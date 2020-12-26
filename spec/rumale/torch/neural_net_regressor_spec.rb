# frozen_string_literal: true

require 'numo/narray'
require 'rumale/dataset'

RSpec.describe Rumale::Torch::NeuralNetRegressor do
  let(:x) do
    centers = Numo::DFloat[[-5, 0], [5, 0]]
    Rumale::Dataset.make_blobs(300, centers: centers, cluster_std: 0.5, random_seed: 1).first
  end

  context 'when single target problem' do
    let(:y) { x[true, 0] + x[true, 1]**2 }

    let(:regressor) do
      class MyNet < Torch::NN::Module
        def initialize
          super
          @dropout2 = Torch::NN::Dropout2d.new(p: 0.5)
          @fc1 = Torch::NN::Linear.new(2, 64)
          @fc2 = Torch::NN::Linear.new(64, 1)
        end

        def forward(x)
          x = @fc1.call(x)
          x = Torch::NN::F.relu(x)
          x = @dropout2.call(x)
          @fc2.call(x)
        end
      end
      model = MyNet.new.to(Torch.device('cpu'))
      described_class.new(model: model, batch_size: 20, max_epochs: 20)
    end

    it do
      Torch.manual_seed 10
      regressor.fit(x, y)
      puts(format('R-Score: %.3f', regressor.score(x, y)))
    end
  end

  context 'when multiple target problem' do
    let(:y) { Numo::DFloat[x[true, 0].to_a, (x[true, 1]**2).to_a].transpose.dot(Numo::DFloat[[0.6, 0.4], [0.8, 0.2]]) }

    let(:regressor) do
      class MyNet < Torch::NN::Module
        def initialize
          super
          @dropout2 = Torch::NN::Dropout2d.new(p: 0.5)
          @fc1 = Torch::NN::Linear.new(2, 64)
          @fc2 = Torch::NN::Linear.new(64, 2)
        end

        def forward(x)
          x = @fc1.call(x)
          x = Torch::NN::F.relu(x)
          x = @dropout2.call(x)
          @fc2.call(x)
        end
      end
      model = MyNet.new.to(Torch.device('cpu'))
      described_class.new(model: model, batch_size: 20, max_epochs: 20)
    end

    it do
      Torch.manual_seed 10
      regressor.fit(x, y)
      puts(format('R-Score: %.3f', regressor.score(x, y)))
    end
  end
end
