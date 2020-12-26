# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/regressor'
require 'rumale/model_selection/shuffle_split'

module Rumale
  module Torch
    class NeuralNetRegressor
      include Base::BaseEstimator
      include Base::Regressor

      def initialize(model:, device: nil, optimizer: nil, loss: nil,
                     batch_size: 50, shuffle: true, max_epochs: 5, validation_split: 0.1,
                     verbose: true)
        @params = method(:initialize).parameters.each_with_object({}) { |(_, kwd), obj| obj[kwd] = binding.local_variable_get(kwd) }
        @params[:device] ||= ::Torch.device('cpu')
        @params[:optimizer] ||= ::Torch::Optim::Adam.new(model.parameters)
        @params[:loss] ||= ::Torch::NN::MSELoss.new
        @params.each_key do |name|
          self.class.send(:define_method, name) { @params[name] }
          self.class.send(:private, name)
        end
      end

      def fit(x, y)
        y = y.expand_dims(1) if y.ndim == 1

        splitter = Rumale::ModelSelection::ShuffleSplit.new(
          n_splits: 1, test_size: validation_split, random_seed: 1
        )
        train_ids, test_ids = splitter.split(x).first
        x_train = x[train_ids, true]
        x_test = x[test_ids, true]
        y_train = y[train_ids, true]
        y_test = y[test_ids, true]

        train_loader = torch_data_loader(x_train, y_train)
        test_loader = torch_data_loader(x_test, y_test)

        1.upto(max_epochs) do |epoch|
          train(train_loader)
          next unless verbose

          puts("Epoch: #{epoch}/#{max_epochs}")
          puts(format('loss: %.4f - val_loss: %.4f', evaluate(train_loader), evaluate(test_loader)))
        end

        self
      end

      def predict(x)
        output = ::Torch.no_grad { model.call(::Torch.from_numo(x).to(:float32)) }.numo
        output.shape[1] == 1 ? output[true, 0].dup : output
      end

      private

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
