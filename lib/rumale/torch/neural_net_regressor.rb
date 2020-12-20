# frozen_string_literal: true

require "rumale/base/base_estimator"
require "rumale/base/regressor"

module Rumale
  module Torch
    class NeuralNetRegressor
      include Base::BaseEstimator
      include Base::Regressor

      def initialize(net:, optimizer:, loss:, metrics:); end

      def fit(x, y); end

      def predict(x); end
    end
  end
end
