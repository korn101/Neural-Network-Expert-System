function result = SigmoidDerivative(activation)

    result = 1*(1 - tanh(activation).^2)/2;

end