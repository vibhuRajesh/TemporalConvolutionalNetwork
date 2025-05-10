import java.util.Arrays;

public class Init {
    public static void main(String[] args) {
        //file in src folder
        DataReader temp = new DataReader("src/SPY.csv");
        //column index 3, series of length 32
        double[][] test = temp.makeSeries(3, 32);
        //folds of size 175 train, 25 test
        double[][][] folds = temp.makeFolds(test, 95, 100);
        //for tracking
        int i = 26;
        double[][] input = folds[i];
        int[] answer = temp.Y()[i];



        //Constructor
        Network network = new Network(input, 0, 95, 128, 5, 1, false, null,  0.00005, answer); //0

        network.addConv( ( (Convolution) network.getLast()).getOutput(), 0, 95, 48, 5, 1, false, null,  0.00005); //1

        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 64, 5, 1, false, null,  0.00005); //2 //output passed-----
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 96, 5, 2, false, null,  0.00005); //3 //input passed     | +
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 48, 5, 4, false, null,  0.00005); //4                    |
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 64, 5, 4, false, null,  0.00005); //5
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 96, 5,  4, false, null,  0.00005); //6 (output passed as residual)
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 64, 5, 4, false, null,  0.00005); //7 //input passed as residual
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 96, 5, 8, false, null,  0.00005); //8
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 128, 5, 8, true, 3,  0.00005); //9


        //Res Block 3
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 48, 5, 2, false, null,  0.00005); //10 (output passed as residual)
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 48, 5, 2, false, null,  0.00005); //11//input passed as residual
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 48, 5, 2, false, null,  0.00005); //12
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 64,5, 4, false, null,  0.00005); //13
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 64, 5, 4, false, null,  0.00005); //14
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 96, 5, 4, false, null,  0.00005); //15
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 128, 5, 8, false, null,  0.00005); //16
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 128, 5, 8, true, 11,  0.00005); //17

        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 48, 5, 2, false, null,  0.00005); //18
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 64, 5, 4, false, null,  0.00005); //19
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 96, 5, 8, false, null,  0.00005); //20
        network.addConv(( (Convolution) network.getLast()).getOutput(), 0, 95, 128, 5, 8, true, 19,  0.00005); //21


        //Last Layer
        network.addFCN(( ( Convolution) network.getLast()).getOutput(), 1, 0.00005, 0, 95); //18




        Network newNetwork = Network.copy((Convolution) network.getFirst(), 0, 95);

        network.train(200, "init", false);


    }
}
