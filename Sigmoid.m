function result = Sigmoid(activation)

    %result = 1./(1+exp(-activation));
    result = 1*(tanh(activation) + 1)/2;

end