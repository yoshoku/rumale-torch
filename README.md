# Rumale::Torch

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

TODO: Write usage instructions here


## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/yoshoku/rumale-torch. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [code of conduct](https://github.com/yoshoku/rumale-torch/blob/master/CODE_OF_CONDUCT.md).

## License

The gem is available as open source under the terms of the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).

## Code of Conduct

Everyone interacting in the Rumale::Torch project's codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/yoshoku/rumale-torch/blob/master/CODE_OF_CONDUCT.md).
