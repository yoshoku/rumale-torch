# frozen_string_literal: true

require_relative 'lib/rumale/torch/version'

Gem::Specification.new do |spec|
  spec.name        = 'rumale-torch'
  spec.version     = Rumale::Torch::VERSION
  spec.authors     = ['yoshoku']
  spec.email       = ['yoshoku@outlook.com']

  spec.summary     = <<~MSG
    Rumale::Torch provides the learning and inference by the neural network defined in torch.rb with the same interface as Rumale
  MSG
  spec.description = <<~MSG
    Rumale::Torch provides the learning and inference by the neural network defined in torch.rb with the same interface as Rumale
  MSG
  spec.homepage    = 'https://github.com/yoshoku/rumale-torch'
  spec.license     = 'BSD-3-Clause'

  spec.metadata['homepage_uri'] = spec.homepage
  spec.metadata['source_code_uri'] = spec.homepage
  spec.metadata['changelog_uri'] = 'https://github.com/yoshoku/rumale-torch/blob/main/CHANGELOG.md'
  spec.metadata['documentation_uri'] = 'https://yoshoku.github.io/rumale-torch/doc/'
  spec.metadata['rubygems_mfa_required'] = 'true'

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  spec.files = Dir.chdir(File.expand_path(__dir__)) do
    `git ls-files -z`.split("\x0").reject { |f| f.match(%r{\A(?:test|spec|features)/}) }
                     .select { |f| f.match(/\.(?:rb|rbs|md|txt)$/) }
  end
  spec.bindir        = 'exe'
  spec.executables   = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ['lib']

  spec.add_dependency 'rumale-core', '>= 0.24'
  spec.add_dependency 'rumale-model_selection', '>= 0.24'
  spec.add_dependency 'rumale-preprocessing', '>= 0.24'
  spec.add_dependency 'torch-rb'
end
