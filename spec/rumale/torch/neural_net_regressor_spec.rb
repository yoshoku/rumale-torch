# frozen_string_literal: true

RSpec.describe Rumale::Torch::NeuralNetRegressor do
  let(:x) do
    centers = Numo::DFloat[[-5, 0], [5, 0]]
    Rumale::Dataset.make_blobs(300, centers: centers, cluster_std: 0.5, random_seed: 1).first
  end

  let(:n_samples) { x.shape[0] }
  let(:verbose) { false }
  let(:validation_split) { 0.1 }
  let(:regressor) do
    described_class.new(
      model: model, batch_size: 20, max_epoch: 20, validation_split: validation_split, verbose: verbose
    ).fit(x, y)
  end
  let(:predicted) { regressor.predict(x) }
  let(:score) { regressor.score(x, y) }
  let(:copied) { Marshal.load(Marshal.dump(regressor)) }

  before { Torch.manual_seed 1 }

  context 'when single target problem' do
    let(:y) { x[true, 0] + x[true, 1]**2 }

    let(:model) do
      Class.new(Torch::NN::Module) do
        def initialize
          super
          @dropout = Torch::NN::Dropout.new(p: 0.1)
          @fc1 = Torch::NN::Linear.new(2, 128)
          @fc2 = Torch::NN::Linear.new(128, 1)
        end

        def forward(x)
          x = @fc1.call(x)
          x = Torch::NN::F.relu(x)
          x = @dropout.call(x)
          @fc2.call(x)
        end
      end.new.to(Torch.device('cpu'))
    end

    it 'learns the model for single target variable problem', :aggregate_failures do
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(score).to be_within(0.02).of(1.0)
    end

    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(copied.params).to eq(regressor.params)
      expect(copied.model).to be_nil
      expect(copied.device).to be_nil
      expect(copied.optimizer).to be_nil
      expect(copied.loss).to be_nil
    end

    context 'when verbose is "true"' do
      let(:verbose) { true }

      it 'outputs debug messages', :aggregate_failures do
        expect { regressor }.to output(/epoch/).to_stdout
      end
    end

    context 'when without validation set' do
      let(:validation_split) { 0 }
      let(:verbose) { true }

      it 'outputs only training dataset loss', :aggregate_failures do
        expect { regressor }.not_to output(/val_loss/).to_stdout
        expect(score).to be_within(0.01).of(1.0)
      end
    end
  end

  context 'when multiple target problem' do
    let(:y) { Numo::DFloat[x[true, 0].to_a, (x[true, 1]**2).to_a].transpose.dot(Numo::DFloat[[0.6, 0.4], [0.8, 0.2]]) }
    let(:n_outputs) { y.shape[1] }

    let(:model) do
      Class.new(Torch::NN::Module) do
        def initialize
          super
          @dropout2 = Torch::NN::Dropout2d.new(p: 0.1)
          @fc1 = Torch::NN::Linear.new(2, 128)
          @fc2 = Torch::NN::Linear.new(128, 2)
        end

        def forward(x)
          x = @fc1.call(x)
          x = Torch::NN::F.relu(x)
          x = @dropout2.call(x)
          @fc2.call(x)
        end
      end.new.to(Torch.device('cpu'))
    end

    it 'learns the model for multi-target variable problem', :aggregate_failures do
      expect(predicted.class).to eq(Numo::DFloat)
      expect(predicted.ndim).to eq(2)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted.shape[1]).to eq(n_outputs)
      expect(score).to be_within(0.02).of(1.0)
    end
  end
end
