# Rumale::Torch

[![Build Status](https://github.com/yoshoku/rumale-torch/workflows/build/badge.svg)](https://github.com/yoshoku/rumale-torch/actions?query=workflow%3Abuild)
[![Gem Version](https://badge.fury.io/rb/rumale-torch.svg)](https://badge.fury.io/rb/rumale-torch)
[![BSD 3-Clause License](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://github.com/yoshoku/rumale-torch/blob/main/LICENSE.txt)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://yoshoku.github.io/rumale-torch/doc/)

Rumale::Torch provides the learning and inference by the neural network defined in [torch.rb](https://github.com/ankane/torch.rb)
with the same interface as [Rumale](https://github.com/yoshoku/rumale).

## Installation
torch.rb is a runtime dependent gem of Rumale::Torch. It requires to install [LibTorch](https://github.com/ankane/torch.rb#libtorch-installation):

    $ brew install automake libtorch

Here, automake is needed to install [rice](https://github.com/jasonroelofs/rice) gem, which torch.rb depends on.

Add this line to your application's Gemfile:

```ruby
gem 'rumale-torch'
```

And then execute:

    $ bundle install

Or install it yourself as:

    $ gem install rumale-torch

## Usage

### Example 1. Pendigits dataset classification

We start by downloading the pendigits dataset from [LIBSVM Data](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) web site.

```bash
$ wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits
$ wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t
```

Training phase:

```ruby
require 'rumale'
require 'rumale/torch'

Torch.manual_seed(1)
device = Torch.device('cpu')

# Loading pendigits dataset consisting of
# 16-dimensional data divided into 10 classes.
x, y = Rumale::Dataset.load_libsvm_file('pendigits')

# Define a neural network in torch.rb framework.
class MyNet < Torch::NN::Module
  def initialize
    super
    @dropout = Torch::NN::Dropout.new(p: 0.5)
    @fc1 = Torch::NN::Linear.new(16, 128)
    @fc2 = Torch::NN::Linear.new(128, 10)
  end

  def forward(x)
    x = @fc1.call(x)
    x = Torch::NN::F.relu(x)
    x = @dropout.call(x)
    x = @fc2.call(x)
    Torch::NN::F.softmax(x)
  end
end

net = MyNet.new.to(device)

# Create a classifier with neural network model.
classifier = Rumale::Torch::NeuralNetClassifier.new(
  model: net, device: device,
  batch_size: 10, max_epoch: 50, validation_split: 0.1,
  verbose: true
)

# Learning classifier.
classifier.fit(x, y)

# Saving model.
Torch.save(net.state_dict, 'pendigits.pth')
File.binwrite('pendigits.dat', Marshal.dump(classifier))
```

Testing phase:

```ruby
require 'rumale'
require 'rumale/torch'

# Loading neural network model.
class MyNet < Torch::NN::Module
  def initialize
    super
    @dropout = Torch::NN::Dropout.new(p: 0.5)
    @fc1 = Torch::NN::Linear.new(16, 128)
    @fc2 = Torch::NN::Linear.new(128, 10)
  end

  def forward(x)
    x = @fc1.call(x)
    x = Torch::NN::F.relu(x)
    # x = @dropout.call(x)
    x = @fc2.call(x)
    Torch::NN::F.softmax(x)
  end
end

net = MyNet.new
net.load_state_dict(Torch.load('pendigits.pth'))

# Loading classifier.
classifier = Marshal.load(File.binread('pendigits.dat'))
classifier.model = net

# Loading test dataset.
x_test, y_test = Rumale::Dataset.load_libsvm_file('pendigits.t')

# Predict labels of test data.
p_test = classifier.predict(x_test)

# Evaluate predicted result.
accuracy = Rumale::EvaluationMeasure::Accuracy.new.score(y_test, p_test)
puts(format("Accuracy: %2.1f%%", accuracy * 100))
```

The result of executing the above scripts is as follows:

```sh
$ ruby train.rb
epoch:  1/50 - loss: 0.2073 - accuracy: 0.3885 - val_loss: 0.2074 - val_accuracy: 0.3853
epoch:  2/50 - loss: 0.1973 - accuracy: 0.4883 - val_loss: 0.1970 - val_accuracy: 0.4893
epoch:  3/50 - loss: 0.1962 - accuracy: 0.4997 - val_loss: 0.1959 - val_accuracy: 0.5013

...

epoch: 50/50 - loss: 0.1542 - accuracy: 0.9199 - val_loss: 0.1531 - val_accuracy: 0.9293

$ ruby test.rb
Accuracy: 91.2%
```

### Example 2. Cross-validation with Rumale

Perform 5-fold cross-validation for regression problem using the housing dataset.

```sh
$ wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing
```

The example script:

```ruby
require 'rumale'
require 'rumale/torch'

Torch.manual_seed(1)
device = Torch.device('cpu')

# Loading pendigits dataset consisting of
# 13-dimensional data with single target variable.
x, y = Rumale::Dataset.load_libsvm_file('housing')

# Define a neural network in torch.rb framework.
class MyNet < Torch::NN::Module
  def initialize
    super
    @fc1 = Torch::NN::Linear.new(13, 128)
    @fc2 = Torch::NN::Linear.new(128, 1)
  end

  def forward(x)
    x = @fc1.call(x)
    x = Torch::NN::F.relu(x)
    x = @fc2.call(x)
  end
end

net = MyNet.new.to(device)

# Create a regressor with neural network model.
regressor = Rumale::Torch::NeuralNetRegressor.new(
  model: net, device: device, batch_size: 10, max_epoch: 100
)

# Create evaluation measure, splitting strategy, and cross validation.
ev = Rumale::EvaluationMeasure::R2Score.new
kf = Rumale::ModelSelection::ShuffleSplit.new(n_splits: 5, test_size: 0.1, random_seed: 1)
cv = Rumale::ModelSelection::CrossValidation.new(estimator: regressor, splitter: kf, evaluator: ev)

# Perform 5-cross validation.
report = cv.perform(x, y)

# Output result.
mean_score = report[:test_score].sum / kf.n_splits
puts(format("5-CV R2-score: %.3f", mean_score))
```

The execution result is as follows:

```ruby
$ ruby cv.rb
5-CV R2-score: 0.755
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/yoshoku/rumale-torch.
This project is intended to be a safe, welcoming space for collaboration,
and contributors are expected to adhere to the [Contributor Covenant](https://contributor-covenant.org) code of conduct.

## License

The gem is available as open source under the terms of the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
