# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/regressor'
require 'rumale/model_selection/shuffle_split'

module Rumale
  module Torch
    # NeuralNetRegressor is a class that provides learning and inference by the neural network defined in torch.rb
    # with an interface similar to regressor of Rumale.
    #
    # @example
    #   require 'rumale/torch'
    #
    #   class MyNet < Torch::NN::Module
    #     def initialize
    #       super
    #       @dropout = Torch::NN::Dropout.new(p: 0.5)
    #       @fc1 = Torch::NN::Linear.new(2, 64)
    #       @fc2 = Torch::NN::Linear.new(64, 1)
    #     end
    #
    #     def forward(x)
    #       x = @fc1.call(x)
    #       x = Torch::NN::F.relu(x)
    #       x = @dropout.call(x)
    #       @fc2.call(x)
    #     end
    #   end
    #
    #   device = Torch.device('gpu')
    #   net = MyNet.new.to(device)
    #
    #   regressor = Rumale::Torch::NeuralNetRegressor.new(model: net, device: device, batch_size: 50, max_epoch: 10)
    #   regressor.fit(x, y)
    #
    #   regressor.predict(x)
    #
    class NeuralNetRegressor < Rumale::Base::Estimator
      include Rumale::Base::Regressor

      # Return the neural nets defined with torch.rb.
      # @return [Torch::NN::Module]
      attr_accessor :model

      # Return the compute device.
      # @return [Torch::Device]
      attr_accessor :device

      # Return the optimizer.
      # @return [Torch::Optim]
      attr_accessor :optimizer

      # Return the loss function.
      # @return [Torch::NN]
      attr_accessor :loss

      # Create a new regressor with neural nets defined by torch.rb.
      #
      # @param model [Torch::NN::Module] The neural nets defined with torch.rb.
      # @param device [Torch::Device/Nil] The compute device to be used.
      #   If nil is given, it to be set to Torch.device('cpu').
      # @param optimizer [Torch::Optim/Nil] The optimizer to be used to optimize the model.
      #   If nil is given, it to be set to Torch::Optim::Adam.
      # @param loss [Torch:NN] The loss function to be used to optimize the model.
      #   If nil is given, it to be set to Torch::NN::MSELoss.
      # @param batch_size [Integer] The number of samples per batch to load.
      # @param max_epoch [Integer] The number of epochs to train the model.
      # @param shuffle [Boolean] The flag indicating whether to shuffle the data at every epoch.
      # @param validation_split [Float] The fraction of the training data to be used as validation data.
      # @param verbose [Boolean] The flag indicating whether to output loss during epoch.
      # @param random_seed [Integer/Nil] The seed value using to initialize the random generator for data splitting.
      def initialize(model:, device: nil, optimizer: nil, loss: nil,
                     batch_size: 128, max_epoch: 10, shuffle: true, validation_split: 0,
                     verbose: false, random_seed: nil)
        super()
        @model = model
        @device = device || ::Torch.device('cpu')
        @optimizer = optimizer || ::Torch::Optim::Adam.new(model.parameters)
        @loss = loss || ::Torch::NN::MSELoss.new
        @params = {}
        @params[:batch_size] = batch_size
        @params[:max_epoch] = max_epoch
        @params[:shuffle] = shuffle
        @params[:validation_split] = validation_split
        @params[:verbose] = verbose
        @params[:random_seed] = random_seed || srand
        define_parameter_accessors
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The target values to be used for fitting the model.
      # @return [NeuralNetRegressor] The learned regressor itself.
      def fit(x, y)
        y = y.expand_dims(1) if y.ndim == 1

        train_loader, test_loader = prepare_dataset(x, y)

        model.children.each do |layer|
          layer.reset_parameters if layer.class.method_defined?(:reset_parameters)
        end

        1.upto(max_epoch) do |epoch|
          train(train_loader)
          display_epoch(train_loader, test_loader, epoch) if verbose
        end

        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) The predicted values per sample.
      def predict(x)
        output = Numo::DFloat.cast(::Torch.no_grad { model.call(::Torch.from_numo(x).to(:float32)) }.numo)
        output.shape[1] == 1 ? output[true, 0].dup : output
      end

      # @!visibility private
      def marshal_dump
        { params: @params }
      end

      # @!visibility private
      def marshal_load(obj)
        @model = nil
        @device = nil
        @optimizer = nil
        @loss = nil
        @params = obj[:params]
        define_parameter_accessors
      end

      private

      def define_parameter_accessors
        @params.each_key do |name|
          self.class.send(:define_method, name) { @params[name] }
          self.class.send(:private, name)
        end
      end

      def prepare_dataset(x, y)
        n_validations = (validation_split * x.shape[0]).ceil.to_i
        return [torch_data_loader(x, y), nil] unless n_validations.positive?

        splitter = Rumale::ModelSelection::ShuffleSplit.new(
          n_splits: 1, test_size: validation_split, random_seed: random_seed
        )
        train_ids, test_ids = splitter.split(x).first
        x_train = x[train_ids, true]
        x_test = x[test_ids, true]
        y_train = y[train_ids, true]
        y_test = y[test_ids, true]
        [torch_data_loader(x_train, y_train), torch_data_loader(x_test, y_test)]
      end

      def torch_data_loader(x, y)
        x_tensor = ::Torch.from_numo(x).to(:float32)
        y_tensor = ::Torch.from_numo(y).to(:float32)
        dataset = ::Torch::Utils::Data::TensorDataset.new(x_tensor, y_tensor)
        ::Torch::Utils::Data::DataLoader.new(dataset, batch_size: batch_size, shuffle: shuffle)
      end

      def train(data_loader)
        model.train
        data_loader.each_with_index do |(data, target), _batch_idx|
          data = data.to(device)
          target = target.to(device)
          optimizer.zero_grad
          output = model.call(data)
          ls = loss.call(output, target)
          ls.backward
          optimizer.step
        end
      end

      def display_epoch(train_loader, test_loader, epoch)
        # rubocop:disable Lint/FormatParameterMismatch
        if test_loader.nil?
          puts(format("epoch: %#{max_epoch.to_s.length}d/#{max_epoch} - loss: %.4f", epoch, evaluate(train_loader)))
        else
          puts(format("epoch: %#{max_epoch.to_s.length}d/#{max_epoch} - loss: %.4f - val_loss: %.4f",
                      epoch, evaluate(train_loader), evaluate(test_loader)))
        end
        # rubocop:enable Lint/FormatParameterMismatch
      end

      def evaluate(data_loader)
        model.eval
        mean_loss = 0
        ::Torch.no_grad do
          data_loader.each do |data, target|
            data = data.to(device)
            target = target.to(device)
            output = model.call(data)
            mean_loss += loss.call(output, target).item
          end
        end

        mean_loss / data_loader.dataset.size
      end
    end
  end
end
