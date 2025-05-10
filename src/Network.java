import java.util.ArrayList;
import java.util.Arrays;
import java.io.Serializable;

public class Network implements Serializable{


    //treat like Linked List:
    private int length ;

    //keep track of size
    private int numConv = 0;
    private int numFCN;
    private FCN firstFCN;

    //"head" of list
    private Layer first;
    //while building network, last could be Convolution or FCN
    private Layer last;

    private ArrayList<FCN> fcnLayers;

    //ground truth
    private int[] answer;

    //for saving state of best model
    //initialized in this.train() based on size of model being trained & epochs

//ArrayList used whenever dimensions need to be flexible/mismatched
    //3D arrays within ArrayList: 1 layer's weights (different sizes so ArrayList utilized) (m: numFilters x  w:windowSize x  p: seriesLength)
    //ArrayList: Network's set of convolution weights
    private ArrayList<double[][][]> convWeights = new ArrayList<>();

    //2D arrays within ArrayList: 1 layer's 1x1 convolution weights)
    //ArrayList: Network's set of all weights
    private ArrayList<double[][]> oWeights = new ArrayList<>();

    //(each 2D array is 1 layer's weights (dimension mismatch)
    //ArrayList: Network's set of FCN weights
    //array of ArrayLists: set of epoch's weights (Network state)
    private ArrayList<double[][]> fcnWeights = new ArrayList<>();

    //1D array: 1 Layer's bias
    //ArrayList: Network's set of bias
    //array of ArrayLists: set of bias across epochs
    private ArrayList<double[]> convBias = new ArrayList<>();
    private ArrayList<double[]> fcnBias = new ArrayList<>();

    //RMSprop tensors for convolution
    private ArrayList<double[][][]> vFilter;
    private ArrayList<double[]> vConvBias = new ArrayList<>();
    private ArrayList<double[][]>  vOne = new ArrayList<>();

    //RMSprop tensors for fully connected layers
    private ArrayList<double[][]> vWeights = new ArrayList<>();
    private ArrayList<double[]> vfcnBias = new ArrayList<>();

    //also initialized in this.train() based on number of epochs passed in for training
    private double[] trainLoss;


    private double[] trainAcc;

    //confusion matrix for precision, recall and f1
    //conf[][0] = TruePos, conf[][1] = TrueNeg, conf[][2] = FalsePos, conf[][3] = FalseNeg          ([] = epoch num)
    private double[][] conf;

    private double[] precision;
    private double[] recall;
    private double[] f1;


    private double testLoss;
    private double testAcc;

    //CONSTRUCTORS
    //1) Empty network
    //2) Premade layer passed in
    //3) Manually instantiate first layer

    /**
     * default constructor for empty network
     */
    public Network(){
        this.length = 0;
    }

    /**
     * overloaded constructor for initializing with head
     * @param first: Convolution layer to set as first
     */
    public Network(Convolution first, int s, int st){
        this.first = first;
        this.last = this.first;
        this.length++;
        this.numConv++;
        //keeps track for reinstantiating with proper window size in validation
        fcnLayers = new ArrayList<>();
    }

    /**
     * Manually instantiate the first layer
     * @param i: input for layer (output from DataReader is 2D)
     * @param s: window start
     * @param st: window stop
     * @param n: number of filters to apply
     * @param k: kernel size (size of each filter)
     * @param d: dilation factor
     * @param r: if layer takes in a residual input from previous layer (false, just needed in Convolution)
     * @param re: residual to be added (null, same as above)
     * @param l: learning rate
     * @param a: answer array (since first layer)
     */
    public Network( double[][] i, int s, int st, int n, int k, int d, boolean r, Convolution re, double l, int[] a ){
        this.answer = a;
        this.first = new Convolution(i, s, st, n, k, d, r, re, l, a);
        this.last = this.first;
        this.length++;
        this.numConv++;
        fcnLayers = new ArrayList<>();

    }
    public Network( double[][][] i, int s, int st, int n, int k, int d, boolean r, Convolution re, double l, int[] a ){
        this.answer = a;
        this.first = new Convolution(i, s, st, n, k, d, r, re, l, a);
        this.last = this.first;
        this.length++;
        this.numConv++;
        fcnLayers = new ArrayList<>();

    }



    //Adding layers: have option to do manually with overloaded methods, or with previously instantiated layer passed in

    //each of the constructors have method, one of Convolution's constructor's is Network constructor
        //Convolution:
            //3D input: standard Conv layer
        //FCN
            //FCN: simply passed in
            //3D input: first after convolution
            //2D input (all following after 3D)

    public void addConv(Convolution toAdd){

        //empty network constructor used
        if(this.length == 0){
            this.first = toAdd;
        }

        else if(this.length == 1){
            //Layer.connect will assign toAdd's .isLast() to true
            Layer.connect(this.first, toAdd);
            //Makes sure value matches in Network
        }
        else{
            Layer.connect(this.last, toAdd);
        }

        //increment length
        this.last = toAdd;
        this.length++;
        this.numConv++;
    }

    //typically used for non-residual layers (can be used if Convolution re is pre instantiated
    public void addConv(double[][][] i, int s, int st, int n, int k, int d, boolean r, Convolution re, double l){
        Convolution toAdd = new Convolution(i, s, st, n, k, d, r, re, l);
        this.addConv(toAdd);
    }

    /**
     * more used for residual layers (only called when residual needed)
     * @param re: index of layer to take residual input from
     */
    public void addConv(double[][][] i, int s, int st, int n, int k, int d, boolean r, int re, double l){
        Convolution res;
        if(r){
            res = (Convolution) (this.get(re));
        }
        else{
            res = null;
        }

        Convolution toAdd = new Convolution(i, s, st, n, k, d, r, res, l);
        this.addConv(toAdd);
    }



    /**
     * PreInstantiated FCN layer passed in
     * @param toAdd: FCN layer to add
     */
    public void addFCN(FCN toAdd){
        if(this.length == 0){
            this.first = toAdd;
        }

        else if(this.length == 1){
            //Layer.connect will assign toAdd's .isLast() to true
            Layer.connect(this.first, toAdd);
            //Makes sure value matches in Network
        }
        else{
            Layer.connect(this.last, toAdd);
        }

        this.last = toAdd;
        this.length++;
        this.numFCN++;

    }

    /**
     * 2D FCN Layer to be added
     * @param i: 2D input array
     * @param o: output neurons (num neurons)
     * @param l: learning rate
     */
    public void addFCN(double[][] i, int o, double l){
        FCN toAdd = new FCN(i, o, l);
        this.addFCN(toAdd);
    }

    /**
     * 3D FCN Layer to be added
     * @param s: start idx
     * @param st: stop idx
     */
    public void addFCN(double[][][] i, int o, double l, int s, int st){
        FCN toAdd = new FCN(i, o, l, s, st);
        this.addFCN(toAdd);
        this.firstFCN = toAdd;
    }
    public static void saveNetwork(Network n, String f){

    }


    /**
     * Training method, calculates accuracy, precision, recall and f1, saves state based on highest f1 score
     * @param epochs: number of epochs to train for
     * @param folder: folder to save each set of weight/bias/ tensors
     * @param save: whether to save file or not
     *
     * @implNote File Names:  <br> 'convWeights.ser' <br> 'fcnWeights.ser' <br> 'oWeights.ser' <br>'convBias.ser' <br>
     *          'fcnBias.ser' <br> <br> Tensors for gradient (RMSprop) <br> 'vConv.ser' <br> 'vfcn.ser' <br> 'vOneWeights.ser' <br>
     *          'vConvBias.ser' <br> 'vfcnBias.ser'
     *
     *
     */
    public void train(int epochs, String folder, boolean save){


        //epoch with highest Accuracy
        double maxAcc = -1;
        Network max = new Network();

        //initialize ArrayList to add Layer info (all based on epoch size)
        //keeps track of model state, initialized within .train()
        //weights


        //keep track of loss/acc/f1 by epoch
        this.trainLoss = new double[epochs];
        this.trainAcc = new double[epochs];

        this.conf = new double[epochs][4];
        this.precision = new double[epochs];
        this.recall = new double[epochs];
        this.f1 = new double[epochs];


        //number of epochs
        for(int i = 0; i < epochs; i++){
            System.out.println("START EPOCH: " + i);

            //forward pass node
            Layer dummy = this.first;

            //make sure  dummy.getNext is != null instead of dummy != null:
                // if dummy == null when exiting loop, null cannot reference .backward() (similar for .backward loop)
            while(dummy.getNext() != null){

                //individual layer step forward
                dummy.forward();
                //move to next layer
                dummy = dummy.getNext();

            }
            dummy.forward();


            //plain accuracy and loss
            this.trainAcc[i] = ( (FCN) dummy).accuracy;
            this.trainLoss[i] = ( (FCN) dummy).loss();

            //maintain reference of maximum accuracy
                        //for precision, recall and f1 score
            double[] guess = ( (FCN) dummy).guess(0.5);
            double tp = 0;
            double fp = 0;
            double tn = 0;
            double fn = 0;


            for(int j = 0;  j < guess.length; j++) {

                //ConfusionMatrix values
                //true, guessed true
                if (this.answer[j] == 1 && guess[j] == 1) {
                    tp++;
                }
                //false, guessed false
                else if (this.answer[j] == 0 && guess[j] == 0) {
                    tn++;
                }
                //false, guessed true
                else if (this.answer[j] == 0 && guess[j] == 1) {
                    fp++;
                }
                //true, guessed false
                else if (this.answer[j] == 1 && guess[j] == 0) {
                    fn++;
                }
            }


            //backward pass (no information needed to store)
            while(dummy.getPrev() != null){

                dummy.backward();
                dummy = dummy.getPrev();

            }
            //same thing as forward: backward pass of first layer
            dummy.backward();


            System.out.println("CONFUSION\n" + this.confusionMatrix(tp, tn, fp, fn));
            //out of all positive guesses, how many were accurate
            this.precision[i] = tp / (tp + fp);

            //out of all in positive class, how many were identified correctly
            this.recall[i] = tp / (tp + fn);

            //(p * r) / (p + r)
            this.f1[i] = 2 * (this.precision[i] * this.recall[i]) / (this.precision[i] + this.recall[i]);


            System.out.println("EPOCH: " + i);
            System.out.println("ACC" + this.trainAcc[i]);
            System.out.println("LOSS" + this.trainLoss[i]);
            System.out.println("ALL ACC " + Arrays.toString(this.trainAcc));
            System.out.println();
            System.out.println("LOSS" + Arrays.toString(this.trainLoss));
            System.out.println("F1" + Arrays.toString(this.f1));

            if(this.trainAcc[i] >= maxAcc){
                maxAcc = this.trainAcc[i];
                max = this;
                System.out.println("MAX " + maxAcc);
                System.out.println("MAX EPOCH " + i);

            }
            System.out.println("\nEND EPOCH: " + i+ "\n--------------------");
        }

        if(save){
            Save.saveNetwork(max, folder);
        }
    }


    /**
     * returns single accuracy value (of epoch)
     * @param guess: predicted value
     * @param truth: ground truth to compare to
     * @return accuracy value
     */
    public double accuracy(double[] guess, int[] truth){
        double acc;

        double count = 0;
        double total = guess.length;

        //guess.length is the window length
        for(int i = 0; i < guess.length; i++){

            //percent that guessed correct class
            if(guess[i] == truth[i]){
                count++;
            }

        }
        return (count / total);
    }


    /**
     * 0-indexed
     * @param index
     * @return Layer at particular index
     */
    public Layer get(int index){
        if(index < this.length && index >= 0){

            Layer temp = this.first;
            while(temp.getIndex() != index){
                temp = temp.getNext();
            }

            return temp;

        }
        else{
            return null;
        }
    }


    public static Network copy(Convolution first, int s, int st){
        Layer iter = first;
        Network copy = new Network(first.getInput(), s, st, first.getNumFilters(), first.getKernel(), first.getDilation(), first.isResid(), null, first.getLearnRate(), first.getAnswer());
        while(iter.getNext() != null){
            iter = iter.getNext();
            //Convolution half of network
            if(iter instanceof Convolution){

                Convolution temp = (Convolution) iter;
                //if not residual layer, then takes in no layer or index as input
                if(!temp.isResid()) {
                    copy.addConv(((Convolution) copy.getLast()).getOutput(), s, st, temp.getNumFilters(), temp.getKernel(), temp.getDilation(), temp.isResid(), null, temp.getLearnRate());
                }

                else{
                    Convolution r = temp.getResid();
                    int idx = r.getIndex();
                    Convolution res = (Convolution) copy.get(idx);
                    copy.addConv(((Convolution) copy.getLast()).getOutput(), s, st, temp.getNumFilters(), temp.getKernel(), temp.getDilation(), temp.isResid(), res, temp.getLearnRate());
                    System.out.println();

                }

            }


            else{
                FCN temp = (FCN) iter;

                if(iter.getPrev() instanceof Convolution){
                    copy.addFCN(((Convolution) copy.getLast()).getOutput(), temp.getNumFilters(), temp.getLearnRate(), s, st);
                }

                else{
                    copy.addFCN(((FCN) copy.getLast()).getOutput(), temp.getNumFilters(), temp.getLearnRate());
                }

            }

        }


        return copy;
    }



    //checker method that isLast() property that Layers have matches Network's last

    public boolean checkLast(){
        return this.last.isLast();
    }
    public Layer getFirst(){
        return this.first;
    }

    public void setRho(double rho){
        Layer iter = this.first;
        while(iter != null){

            if(iter instanceof Convolution){
                ((Convolution) iter).setRho(rho);
            }
            else{
                ((FCN) iter).setRho(rho);
            }
            iter = iter.getNext();


        }
    }

    public String confusionMatrix(double tp, double tn, double fp, double fn){

        String t = "|-----------+-----------+-----------|\n" +
                   "|           | Predicted |           |\n" +
                   "|-----------|-----------|-----------|\n" +
                   "|  Actual   |   1(Up)   |  0(down)  |\n" +
                   "|-----------|-----------|-----------|\n" +
                   "|   1(up)   |TP: " + (int) tp + "     |FN: " + (int)fn + "    |\n" +
                   "|-----------|-----------|-----------|\n" +
                   "|  0(down)  |FP: " + (int)fp + "     |TN: " + (int)tn + "     |\n" +
                   "|-----------|-----------|-----------|\n";


        return t;
    }

    public Layer getLast(){
        return this.last;
    }
    public int getLength(){
        return this.length;
    }

    public double[] getTrainAcc() {
        return trainAcc;
    }

}
