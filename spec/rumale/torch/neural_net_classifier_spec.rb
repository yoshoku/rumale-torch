# frozen_string_literal: true

RSpec.describe Rumale::Torch::NeuralNetClassifier do
  let(:classifier) do
    class MyNet < Torch::NN::Module
      def initialize
        super
        @dropout = Torch::NN::Dropout.new(p: 0.5)
        @fc1 = Torch::NN::Linear.new(2, 8)
        @fc2 = Torch::NN::Linear.new(8, 2)
      end

      def forward(x)
        x = @fc1.call(x)
        x = Torch::NN::F.relu(x)
        x = @dropout.call(x)
        x = @fc2.call(x)
        Torch::NN::F.softmax(x)
      end
    end
    model = MyNet.new.to(Torch.device('cpu'))
    described_class.new(model: model, batch_size: 20, max_epoch: 20)
  end

  let(:dataset) do
    x_a, y_a = Rumale::Dataset.make_blobs(200, centers: Numo::DFloat[[-5, 5], [5,  5]], cluster_std: 0.6,
                                               random_seed: 1)
    x_b, y_b = Rumale::Dataset.make_blobs(200, centers: Numo::DFloat[[5, -5], [-5, -5]], cluster_std: 0.6,
                                               random_seed: 1)
    x = Numo::DFloat.vstack([x_a, x_b])
    y = 2 * y_a.concatenate(y_b) - 1
    [x, y]
  end

  let(:x) { dataset[0] }
  let(:y) { dataset[1] }

  it do
    Torch.manual_seed 10
    classifier.fit(x, y)
    p classifier.predict(x)
    puts(format('Accuracy: %2.1f%%', (classifier.score(x, y) * 100)))
  end
end
