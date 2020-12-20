# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/classifier'
require 'rumale/preprocessing/label_encoder'
require 'rumale/model_selection/function'

module Rumale
  module Torch
    class NeuralNetClassifier
      include Base::BaseEstimator
      include Base::Classifier

      def initialize(model:, device: nil, optimizer: nil, loss: nil,
                     batch_size: 50, shuffle: true, max_epochs: 5, validation_split: 0.1,
                     verbose: true)
        @params = {}
        @params[:model] = model
        @params[:device] = device || ::Torch.device('cpu')
        @params[:optimizer] = optimizer || ::Torch::Optim::Adam.new(model.parameters)
        @params[:loss] = loss || ::Torch::NN::CrossEntropyLoss.new
        @params[:batch_size] = batch_size
        @params[:shuffle] = shuffle
        @params[:max_epochs] = max_epochs
        @params[:validation_split] = validation_split
        @params[:verbose] = verbose
      end

      def fit(x, y)
        encoder = Rumale::Preprocessing::LabelEncoder.new
        encoder.fit(y)
        @classes = Numo::NArray[*encoder.classes]

        x_train, x_test, y_train, y_test = Rumale::ModelSelection.train_test_split(
          x, y, test_size: validation_split, stratify: true, random_seed: 1
        )

        y_train = encoder.transform(y_train)
        y_test = encoder.transform(y_test)

        train_loader = torch_data_loader(x_train, y_train)
        test_loader = torch_data_loader(x_test, y_test)

        1.upto(max_epochs) do |epoch|
          train(train_loader)
          next unless verbose

          puts("Epoch: #{epoch}/#{max_epochs}")
          puts('loss: %.4f - accuracy: %.4f - val_loss: %.4f - val_accuracy: %.4f' % [
            evaluate(train_loader), evaluate(test_loader)
          ].flatten)
        end

        self
      end

      def predict(x)
        output = model.call(::Torch.from_numo(x).to(:float32))
        _, indices = ::Torch.max(output, 1)
        @classes[indices.numo].dup
      end

      def decision_function(x)
        output = model.call(::Torch.from_numo(x).to(:float32))
        output.numo
      end

      private

      def torch_data_loader(x, y)
        x_tensor = ::Torch.from_numo(x).to(:float32)
        y_tensor = ::Torch.from_numo(y).to(:int64)
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
        correct = 0
        ::Torch.no_grad do
          data_loader.each do |data, target|
            data = data.to(device)
            target = target.to(device)
            output = model.call(data)
            mean_loss += loss.call(output, target).item
            pred = output.argmax(1, keepdim: true).view(-1)
            correct += pred.eq(target.view_as(pred)).sum.item
          end
        end

        mean_loss /= data_loader.dataset.size
        accuracy = correct.fdiv(data_loader.dataset.size)
        [mean_loss, accuracy]
      end

      def model
        @params[:model]
      end

      def device
        @params[:device]
      end

      def optimizer
        @params[:optimizer]
      end

      def loss
        @params[:loss]
      end

      def batch_size
        @params[:batch_size]
      end

      def shuffle
        @params[:shuffle]
      end

      def max_epochs
        @params[:max_epochs]
      end

      def validation_split
        @params[:validation_split]
      end

      def verbose
        @params[:verbose]
      end
    end
  end
end
