# frozen_string_literal: true

RSpec.describe Rumale::Torch::NeuralNetClassifier do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:n_samples) { x.shape[0] }
  let(:verbose) { false }
  let(:validation_split) { 0.1 }
  let(:classifier) do
    described_class.new(
      model: model, batch_size: 20, max_epoch: 20, validation_split: validation_split, verbose: verbose
    ).fit(x, y)
  end
  let(:func_vals) { classifier.decision_function(x) }
  let(:predicted) { classifier.predict(x) }
  let(:score) { classifier.score(x, y) }
  let(:copied) { Marshal.load(Marshal.dump(classifier)) }

  before { Torch.manual_seed 1 }

  context 'when binary classification problem' do
    let(:n_classes) { 2 }

    let(:dataset) do
      x_a, y_a = Rumale::Dataset.make_blobs(200, centers: Numo::DFloat[[-5, 5], [5,  5]], cluster_std: 0.6, random_seed: 1)
      x_b, y_b = Rumale::Dataset.make_blobs(200, centers: Numo::DFloat[[5, -5], [-5, -5]], cluster_std: 0.6, random_seed: 1)
      x = Numo::DFloat.vstack([x_a, x_b])
      y = 2 * y_a.concatenate(y_b) - 1
      [x, y]
    end

    let(:model) do
      Class.new(Torch::NN::Module) do
        def initialize
          super
          @dropout = Torch::NN::Dropout.new(p: 0.2)
          @fc1 = Torch::NN::Linear.new(2, 64)
          @fc2 = Torch::NN::Linear.new(64, 2)
        end

        def forward(x)
          x = @fc1.call(x)
          x = Torch::NN::F.relu(x)
          x = @dropout.call(x)
          x = @fc2.call(x)
          Torch::NN::F.softmax(x)
        end
      end.new.to(Torch.device('cpu'))
    end

    it 'classifies two clusters', :aggregate_failures do
      expect(classifier.classes.class).to eq(Numo::Int32)
      expect(classifier.classes.ndim).to eq(1)
      expect(classifier.classes.shape[0]).to eq(n_classes)
      expect(func_vals.class).to eq(Numo::DFloat)
      expect(func_vals.ndim).to eq(2)
      expect(func_vals.shape[0]).to eq(n_samples)
      expect(func_vals.shape[1]).to eq(n_classes)
      expect(predicted.class).to eq(Numo::Int32)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.01).of(1.0)
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(copied.params).to eq(classifier.params)
      expect(copied.classes).to eq(classifier.classes)
      expect(copied.model).to be_nil
      expect(copied.device).to be_nil
      expect(copied.optimizer).to be_nil
      expect(copied.loss).to be_nil
    end

    context 'when verbose is "true"' do
      let(:verbose) { true }

      it 'outputs debug messages', :aggregate_failures do
        expect { classifier }.to output(/epoch/).to_stdout
      end
    end

    context 'when without validation set' do
      let(:validation_split) { 0 }
      let(:verbose) { true }

      it 'outputs only training dataset loss', :aggregate_failures do
        expect { classifier }.not_to output(/val_loss/).to_stdout
        expect(score).to eq(1)
      end
    end
  end

  context 'when multiclass classification problem' do
    let(:n_classes) { 3 }

    let(:dataset) do
      centers = Numo::DFloat[[0, 5], [-5, -5], [5, -5]]
      Rumale::Dataset.make_blobs(300, centers: centers, cluster_std: 0.5, random_seed: 1)
    end

    let(:model) do
      Class.new(Torch::NN::Module) do
        def initialize
          super
          @dropout = Torch::NN::Dropout.new(p: 0.2)
          @fc1 = Torch::NN::Linear.new(2, 64)
          @fc2 = Torch::NN::Linear.new(64, 3)
        end

        def forward(x)
          x = @fc1.call(x)
          x = Torch::NN::F.relu(x)
          x = @dropout.call(x)
          x = @fc2.call(x)
          Torch::NN::F.softmax(x)
        end
      end.new.to(Torch.device('cpu'))
    end

    it 'classifies three clusters', :aggregate_failures do
      expect(classifier.classes.class).to eq(Numo::Int32)
      expect(classifier.classes.ndim).to eq(1)
      expect(classifier.classes.shape[0]).to eq(n_classes)
      expect(func_vals.class).to eq(Numo::DFloat)
      expect(func_vals.ndim).to eq(2)
      expect(func_vals.shape[0]).to eq(n_samples)
      expect(func_vals.shape[1]).to eq(n_classes)
      expect(predicted.class).to eq(Numo::Int32)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.01).of(1.0)
    end
  end
end
