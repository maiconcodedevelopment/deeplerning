import numpy
import matplotlib.pyplot as plot
import pandas as pd

if __name__ == '__main__':

    data = pd.DataFrame(data=[['sp','saopaulo','Sao Paulo'],['pe','pernabumco','recife'],['rj','rio de janeiro','Rio de Janeiro']],columns=['Simgla','None','Capital'])
    data.index = pd.Index([0,1,2])

    s = str('letsin')
    st = ["_" for sa in s]
    stc = [x for x in st if x * 1 ]
    print(stc)


    #n_cols = data.shape[1]
    #model = Sequential()
    #model.add(Dense(50,activation='relu',input_shape=(n_cols,)))
    #model.add(Dense(32,activation='relu'))
    #model.add(Dense(1))
    #model.compile(optimizer='adam',loss='mean_squared_error')
    #model.fit()

    def relu(input):
        output = max(0,input)
        return output

    def prect_previre(input,weights):

        a1 = (input * weights['x1']).sum()
        a1_out = relu(a1);print(a1_out)
        a2 = (input * weights['x2']).sum()
        a2_out = relu(a2);print(a2_out)

        resultout = numpy.array([a1_out,a2_out])

        a3 = (resultout * weights['x3']).sum()
        a3_out = relu(a3);print(a3_out)

        a4 = (resultout * weights['x4']).sum()
        a4_out = relu(a4);print(a4_out)

        outa3a4 = numpy.array([a3_out,a4_out])

        outputresult = (outa3a4 * weights['out']).sum()

        return relu(outputresult)

    def get_slope(input_data,target,weights):
        a = (weights * input_data).sum()
        error = a - target
        slope = 2 * input_data * error
        return slope

    def get_mse(input_data,target,weights):
        a = (weights * input_data).sum()
        error = a - target
        return error

    node = {
        'x1':numpy.array([2,4]),
        'x2':numpy.array([4,-5]),
        'x3':numpy.array([-1,2]),
        'x4':numpy.array([2,1]),
        'out':numpy.array([1,1])
    }

    node1 = {
        'x1': numpy.array([2, 4]),
        'x2': numpy.array([4, -5]),
        'x3': numpy.array([-1, 2]),
        'x4': numpy.array([1, 2]),
        'out': numpy.array([1, 1])
    }

    target = 3
    weights = numpy.array([0,2,1])
    input_data = numpy.array([1,2,3])
    numb = 20
    ap = []

    for i in range(numb):

        slope = get_slope(input_data,target,weights)

        weights = weights - 0.01 * slope

        mse = get_mse(input_data,target,weights)
        print(slope,weights,mse)
        ap.append(mse)


    plot.plot(ap)
    plot.xlabel('Iterations')
    plot.ylabel('mean squared error')
    plot.show()

    exit()

    # Import necessary modules
    import keras