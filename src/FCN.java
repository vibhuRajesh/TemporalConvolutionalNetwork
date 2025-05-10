import java.lang.Math;
import java.util.Arrays;
import java.util.Random;

public class FCN extends Layer {

    //inherited
    private Layer prev;
    private Layer next;
    private boolean isFirst;
    private boolean isLast;

    private double learnRate;

    //batch of flattened arrays
    private double[][] input;
    private double[][] output;


    //each column is one output neuron's set of corresponding weights
    private double[][] weights;
    private double[] bias;
    private double[] guess;


    //velocity tensors and rho: constant in RMSprop (impacts how strongly past gradients affect moving average)
    private double[][] vweights;
    private double[] vbias;

    private double rho;
    //regularization constant and weight penalty
    private double lambda;
    //elements

    //optional instance variables if previous layer is Convolution type
    //start and stop window
    private int s;
    private int st;


    //used in backward pass
    private double[][] preactivate;
    private double[][] prebias;

    //tensors to send back in backpropagation
    //back: previous layer is another FCN and takes a 2D gradient
    //backs: previous layer is a Convolution layer and needs to be deflattened
    private double[][] back;
    private double[][][] backs;

    //if previous layer is Convolution, need to deflatten
    //num filters in previous layer
    private int m;
    //series of length p
    private int p;


    private int[] answer;

    //stores accuracy in last layer
    public double accuracy;

    /** CONSTRUCTOR 1
     * @implNote: randomizes filters at the end
     * @param i: 2D input array
     * @param o: output size (num neurons)
     * @param l: learning rate
     */

    public FCN ( double[][] i, int o, double l) {
        this.input = i;

        //i.length = num training samples
        this.output = new double[i.length][o];
        this.preactivate = new double[i.length][o];
        this.prebias = new double[i.length][o];


        //one bias term per output neuron
            //i[0].length = num input neurons per training sample
        //also includes velocity tensors
        this.weights = new double[o][i[0].length];
        this.vweights = new double[o][i[0].length];

        this.bias = new double[o];
        this.vbias = new double[o];

        this.rho = 0.9;
        this.lambda = 0;

        this.randomizeFilters() ;



    }


    /** CONSTRUCTOR 2
     * @implNote overloaded constructor for 3D array input ( from Convolution Layer )
     * @implNote: flattens into a 2D array of (st - s) 1D arrays of length (channels * seriesLength)
     * @param i: input array as 3D
     * @param o: output size
     * @param l: learning rate
     */
    public FCN ( double[][][] i, int o, double l, int s, int st) {
        int length = i.length;
        int width = i[0][0].length;

        //flatten 3D array
        //num samples x (flattened l x w)
        double[][] temp = new double[st - s][length * width];


        int count = 0;

        //num samples
        //can't call .flatten() because constructor has not been called yet
        for(int j = s; j < st; j++){
            count = 0;

            //length (depth)
            for(int h = 0; h < length; h++){

                //width (length of series)
                for(int k = 0; k < width; k++){
                    temp[j - s][count] = i[h][j][k];
                    count++;
                }

            }
        }

        this ( temp, o, l ) ;
        this.m = length;
        this.p = width;
        int w = i[0].length;


        //anything out of range of st-s for w will be 0
        this.backs = new double[m][w][p];

        this.s = s;
        this.st = st;


    }


    /**
     * @implNote Forward pass through layer
     * @implNote: updates input to previous layer's output after checking layer type
     * @return
     */
    public double[][] forward() {
        this.answer = this.getAnswer();

        //if not first layer, the input should be previous layer's output
        if(!this.isFirst()){

            //Layer.getOutput() returns Object
            //FCN output is 2D array so dimensions match
            if(this.prev instanceof FCN){
                this.input = (double[][]) this.prev.getOutput();
            }

            //Convolution output is 3D array, needs to be flattened
            else if(this.prev instanceof Convolution){
                //flattened each time forward pass
                double[][][] i = (double[][][]) this.prev.getOutput();

                this.input = this.flatten(i , this.s, this.st);
            }

        }

        //normal forward propagation
        this.compute();

        for(int i = 0; i < this.output.length; i++){
            this.prebias[i] = Arrays.copyOf(this.output[i], this.output[i].length);
        }

        this.addBias();
        //last layer uses sigmoid
        if(!this.isLast()) {
            //if not last layer, then copy entire array, if last layer, then each array is 1 element
            for(int i = 0; i < this.output.length; i++){
                this.preactivate[i] = Arrays.copyOf(this.output[i], this.output[i].length);
            }

            this.SeLU();
        }
        else{
            for(int i = 0; i < this.output.length; i++){
                this.preactivate[i][0] = this.output[i][0];
            }

            this.sigmoid();
            this.loss();
        }


        System.out.println("index " + this.getIndex() + "output" + (Arrays.toString(this.output[0])));
        return this.output;
    }

    private void randomizeFilters() {
        Random a = new Random() ;
        Random b = new Random() ;


        for ( int i = 0; i < this.weights.length; i++ ) {
            for ( int j = 0; j < this.weights[0].length; j++ ) {

                //LeCun normal distribution with mean 0, and SD: 1/n: n being inputs in the weight tensor
                //normal distribution * SD + mean
                this.weights[i][j] = a.nextGaussian() *  ( 1.0 / this.input[0].length ) ;
                this.weights[i][j] *= Math.sqrt(this.input[0].length);

            }
            this.bias[i] = b.nextGaussian() *  ( (1.0 / this.input[0].length) );
            this.bias[i] *= Math.sqrt(this.input[0].length);

        }
    }


    /**
     * @implNote flattens along depth dimension [i][][], assumes input is (numFilters x numSamples x seriesLength)
     * @param i: 3D input array
     * @param s: start
     * @param st: stop
     * @return 2D array flattened along DEPTH
     */
    public double[][] flatten(double[][][] i, int s, int st){
        //[windowSize][elements in one sample]
        double[][] temp = new double[st - s][i.length * i[0][0].length];

        int count = 0;

        //num samples
        for(int j = s; j < st; j++){
            count = 0;

            //length (depth)
            for(int h = 0; h < i.length; h++){

                //width (length of series)
                for(int k = 0; k < i[0][0].length; k++){
                    temp[j - s][count] = i[h][j][k];
                    count++;
                }

            }
        }

        return temp;

    }

    /**
     * @implNote connects weights from one layer to previous layers and calculates output
     * @return 2D array with o output nodes for each sample in window
     */
    public double[][] compute(){

        if(this.prev instanceof FCN){
            this.s = ((FCN) this.prev).s;
            this.st = ((FCN) this.prev).st;
        }


        //over each sample in st - s
        for(int h = 0; h < this.input.length; h++ ){
            //over each output node's set of weights
            for(int i = 0; i < this.weights.length; i++){

                double sum = 0;
                //for each element of input layer, and corresponding weight within output node
                for(int j = 0; j < this.input[h].length; j++){
                    sum += this.input[h][j] * this.weights[i][j];

                }
                this.output[h][i] = sum;
            }
        }


        return this.output;
    }

    /**
     * @implNote adds bias, called in this.forward()
     */
    public void addBias(){

        //for each sample series
        for(int i = 0; i < this.output.length; i++){

            //over each node in output
            for(int j = 0; j < this.output[i].length; j++){

                this.output[i][j] += this.bias[j] * 0.01;
            }

        }

    }


    /**
     * @implNote x<= 0: y = lambda * ( alpha * ( e^x - 1 ) ) , if x > 0: y = lambda * x
     * @implNote: alpha = 1.673263242354377284817042991671, lambda = 1.0507009873554804934193349852946 +
     * @implNote: in this.forward()
     */
    public void SeLU(){

        double alpha = 1.673263242354377284817042991671;
        double lambda = 1.0507009873554804934193349852946;

        for(int i = 0; i < this.output.length; i++){
            for(int j = 0; j < this.output[i].length; j++){

                //selu rule
                if ( this.output[i][j] <= 0 ) {
                    this.output[i][j] = lambda * ( alpha * ( Math.exp ( this.output[i][j] ) - 1 ) ) ;
                }

                else{
                    this.output[i][j] = lambda * ( this.output[i][j] ) ;
                }

            }
        }

    }

    /**
     * @implNote used for last layer to get probabilities for binary cross entropy
     * @implNote: important that it be last layer because only operates on first element
     */
    public void sigmoid(){

        //checks if last
        if(this.isLast){
            for (int i = 0; i < this.output.length; i++) {

                //sigmoid: (1 / (1 + (e ^ -x)))
                double temp = ( 1 / ( 1 + Math.exp( -1 * this.output[i][0]) ) );
                this.output[i][0] = temp;
            }
        }


    }

    public double reg(){
        double sum = 0;
        for(int i = 0; i < this.weights.length; i++){
            for(int j = 0; j < this.weights[i].length; j++){


                sum += this.weights[i][j] * this.weights[i][j];
            }
        }

        return sum;
    }


    /**
     * @implNote Binary classifier for all series in batch (s:st)
     * @implNote; checks if last layer
     * @param threshold: threshold for splitting classes across, typically 0.5
     * @return 0 (decrease) or 1 (increase)
     */
    public double[] guess(double threshold){
        //only last layer has format for guessing
        if(!this.isLast){
            return null;
        }

        else{
            this.guess = new double[this.output.length];
            for(int i = 0; i < this.output.length; i++){
                if(this.output[i][0] >= threshold){
                    this.guess[i] = 1;
                }
                else{
                    this.guess[i] = 0;
                }
            }
        }
        return this.guess;

    }

    /**
     *
     * @return loss value across batch
     */
    public double loss(){

        double[] answerRange = new double[this.st - this.s];

        double batchLoss = 0;
        double[] guess = this.guess(0.5);


        double count = 0;
        for(int i = 0; i < this.guess.length; i++){
            answerRange[i] = this.answer[i + this.s];
            if(this.answer[i + this.s] == 1){
                count++;
            }
        }

        double correct = 0;
        double total = guess.length;
        for(int i = 0; i < guess.length; i++){

            batchLoss +=  -1 * ( (this.answer[i + this.s] * Math.log(this.output[i][0]) ) + ( ( 1 - this.answer[i + this.s]) * Math.log(1 - this.output[i][0]) ) );

            if(this.answer[i + this.s] == guess[i]){

                correct++;

            }
        }
        System.out.println("loss output " + Arrays.deepToString(this.output));
        this.accuracy = (correct / total);
        System.out.println("ACC" + this.accuracy);
        System.out.println("GUESS" + Arrays.toString(this.guess));
        System.out.println("ANS" + Arrays.toString(answerRange));
        System.out.println("S" + this.s);
        System.out.println("ST" + this.st);

        System.out.println();
        batchLoss /= (total);
        return batchLoss;

    }


    //BACKWARD PASS

    public void backward(){

        //CASE I: Last Layer
        if(this.isLast()){

            //tensor of gradients wrt Loss function
            // ∂L/∂g where L = BCE(g)
            double[][] grad = this.dLoss();

            //internally calculates local gradient of sigmoid operation
            //multiplies by upstream (grad)
            this.dSigmoid(grad);


            //Learnable Parameters: Bias and Weights
            //updates bias
            this.dBias(grad);
            //updates weights
            this.dWeights(grad);


            //setting backward input for previous layer
            this.setBack(this.dInput(grad));
            if (this.prev instanceof Convolution) {
                this.setBacks(this.deflatten(this.back));
            }

        }

        //CASE II: Not Last
        //CASE III: Not Last and previous is Convolution (deflatten)
        else{
            //next layer's set of gradients
            double[][] grad = this.getBack();

            this.dSeLU(grad);

            this.dBias(grad);

            this.dWeights(grad);

            this.setBack(this.dInput(grad));
            if (this.prev instanceof Convolution) {
                this.setBacks(this.deflatten(this.back));
            }



        }
    }

    /**
     * ADD SEED AND METHOD CALL
     * @implNote: perturbs 1 weight 1 time (either f(x+h) or f(x-h)), manually add method call in Network.train()
     * corresponding to the layer, then run 1 fwd pass
     * @param epsilon: perturbation value (negative for
     * @param idx: array with indices of weight to perturb
     */
    public void gradCheck(double epsilon, int[] idx){
        int i = idx[0];
        int j = idx[1];
        this.weights[i][j] += epsilon;
    }

    /**
     * @implNote guesses are actually probability values output from the sigmoid, for last layer
     * @return 2D tensor of gradients of Loss wrt guesses
     */
    public double[][] dLoss(){

        //tensor to store gradients
        //doesn't return null since only called in last layer
        double[][] grad = new double[this.guess(0.5).length][1];

        for(int i = 0; i < grad.length; i++){

            //yhat = guesses
            double yhat = this.output[i][0];

            //y = ground truth
            double y = this.answer[i];

            grad[i][0] = (yhat - y) / ( yhat * (1 - yhat) );

        }
        return grad;
    }


    /**
     * @param u: upstream gradient: ∂L/∂g
     * @return 2D tensor of downstream gradients
     * @implNote e.g. if u = SeLU(e), calculates: ∂L/∂e = ∂L/∂u * ∂u/∂e
     */
    public double[][] dSeLU(double[][] u){
        double alpha = 1.673263242354377284817042991671;
        double lambda = 1.0507009873554804934193349852946;

        for(int i = 0; i < u.length; i++){
            for(int j = 0; j < u[i].length; j++){


                //if (x > 0): ∂L/∂u *  λ
                if(this.preactivate[i][j] > 0){
                    u[i][j] *= (lambda);
                }

                //if (x <= 0): ∂L/∂u * (λ * α * e^x)
                else{
                    double x = this.preactivate[i][j];
                    u[i][j] *= (lambda * alpha * Math.exp(x));
                }

            }
        }
        return u;
    }

    /**
     * @param u: upstream gradient ∂L/∂g
     * @return 2D tensor of downstream gradients
     * @implNote e.g. if u = σ(e), calculates ∂L/∂e = ∂L/∂u * ∂u/∂e
     */
    public double[][] dSigmoid(double[][] u){

        for(int i = 0; i < u.length; i++){
            //sigmoid formula
            double sgmd = 1 / (1 + Math.exp(-1 * this.preactivate[i][0]));

            //σ`(x) = ( σ(x) * (1 - σ(x)) )
            double grad = (sgmd * (1 - sgmd));

            u[i][0] *= grad;
        }

        return u;
    }

    /**
     * @implNote upstream gradient is typically tensor after dSeLU or dSigmoid
     * @implNote: updates biases as well
     * @param u: upstream gradient
     * @return gradient of bias terms
     */
    public double[] dBias(double[][] u){

        //bias gradient vector is same length as number of output nodes
        double[] grad = new double[u[0].length];

        //each neuron
        for(int i = 0; i < u[0].length; i++){

            double sum = 0;

            //each batch member
            for(int j = 0; j < u.length; j++){
                sum += u[j][i];
            }
            grad[i] = sum;

        }

        this.step(this.bias, grad, this.vbias, this.rho);

        return grad;
    }


    /**
     *
     * @implNote updates weights after calculation
     * @param u: upstream gradient
     * @return gradient wrt weights
     */
    public double[][] dWeights(double[][] u){
        //same dimensions as weights
        double[][] grad = new double[this.weights.length][this.weights[0].length];

        //each output neuron
        for(int i = 0; i < grad.length; i++){

            //each weight
            for(int j = 0; j < grad[0].length; j++){

                double sum = 0;
                //each batch member
                for(int k = 0; k < this.input.length; k++){

                    //1 ele of downstream grad: ∂L/∂b  = ∂L/∂C * ∂C/∂b
                    sum += u[k][i] * this.input[k][j];

                }
                grad[i][j] = sum + (2 * this.lambda * this.weights[i][j]);

            }

        }

        this.step(this.weights, grad, this.vweights, this.rho);
        return grad;
    }

    public double[][] dInput(double[][] u){
        //window size
        int w = this.input.length;
        //num neurons in each input element
        int m = this.input[0].length;
        //num output neurons
        int o = this.weights.length;
        double[][] grad = new double[w][m];

        //each batch member
        for(int i = 0; i < w; i++){
            //each input neuron
            for(int j = 0; j < m; j++){

                //sum to keep track of gradient
                //1 ele of downstream grad: ∂L/∂a  = ∂L/∂C * ∂C/∂a
                double sum = 0;
                //each output neuron
                for (int k = 0; k < o; k++) {

                    //i: corresponding to sample
                    //k: down each output node of upstream
                    sum += u[i][k] * this.weights[k][j];

                }
                grad[i][j] = sum;

            }
        }

        return grad;
    }


    /**
     *
     * @param flatten: 2D array to deflatten
     * @return 3D array: this backs
     */
    public double[][][] deflatten(double[][] flatten){

        //m x w x p to populate only s-st but have full dimensions to send back
        double[][][] t = new double[this.m][this.backs[0].length][this.p];

        //each filter
        for(int j = s; j < st; j++){
            int count = 0;
            //each batch member
            for(int i = 0; i < this.m; i++){
                //each series member
                for(int k = 0; k < this.p; k++ ){

                    t[i][j][k] = flatten[j - s][count];
                    count++;
                }
            }
        }
        return t;

    }

    //optimizer

    /**
     *
     * @param arr: tensor to be updated
     * @param grad: gradient tensor
     * @param velo: velocity tensor
     * @param rho: whether to use momentum (0 if not)
     */

    public void step(double[] arr, double[] grad, double[] velo, double rho){

        //uses momentum
        for(int i = 0; i < grad.length; i++){
            //vt = vt-1 * ρ +( 1 - ρ ) * grad^2
            velo[i] = (rho * velo[i]) + ( (1 - rho) * grad[i]* grad[i]);

            //1E-8 for divide by 0 error
            arr[i] -= (this.learnRate/Math.sqrt(velo[i] + 1E-8) ) * grad[i];
        }

    }


    public void setWeights(double[][] weights){
        this.weights = this.copyArray(weights);
    }
    public void setBias(double[] bias){
        this.bias = this.copyArray(bias);

    }


    public void step(double[][] arr, double[][] grad,double[][] velo, double rho){
        //uses momentum
        for(int i = 0; i < grad.length; i++){
            for(int j = 0; j < grad[0].length; j++){

                //vt = vt-1 * ρ +( 1 - ρ ) * grad^2
                velo[i][j] = (rho * velo[i][j]) + ( (1 - rho) * grad[i][j] * grad[i][j]);

                //1E-8 for divide by 0 error
                arr[i][j] -= this.learnRate/Math.sqrt(velo[i][j] + 1E-8) * grad[i][j];

            }
        }



    }

    //setter for gradient for previous layer to access
    public void setBack(double[][] back){
        this.back = back;
    }

    public void setBacks(double[][][] backs){
        this.backs = backs;
    }


    //to access next layer's gradient passed back
    public double[][] getBack(){
        return ((FCN) this.getNext()).back;
    }
    public double[][][] getBacks(){
        return this.backs;
    }


    //copy 1, and 2D array for manipulation

    public double[][] copyArray(double[][] arr){

        double[][] temp = new double[arr.length][arr[0].length];

        for(int i = 0; i < arr.length; i++){
            for(int j = 0; j < arr[0].length; j++){
                temp[i][j] = arr[i][j];
            }
        }
        return temp;
    }

    public double[] copyArray(double[] arr){

        double[] temp = new double[arr.length];
        for(int i = 0; i < arr.length; i++){
            temp[i] = arr[i];
        }
        return temp;
    }





    @Override
    public void setPrev ( Layer prev ) {
        this.prev = prev;
    }

    @Override
    public void setNext ( Layer next ) {
        this.next = next;
    }

    @Override
    public Layer getPrev() {
        return this.prev;
    }

    @Override
    public Layer getNext() {
        return this.next;
    }

    /**
     * 0-indexed
     * @return Layer's index
     */
    public int getIndex(){
        if(this.prev == null){
            return 0;
        }
        else{
            return this.prev.getIndex() + 1;
        }
    }

    public void setRho(double rho){this.rho = rho;}

    public double getLearnRate(){
        return this.learnRate;
    }
    public int getNumFilters(){
        return this.weights.length;
    }

    @Override


    public boolean isFirst(){

        if(this.prev == null){
            this.isFirst = true;
            return true;
        }
        else{
            this.isFirst = false;
            return false;
        }
    }
    public boolean isLast(){

        if(this.next == null){
            this.isLast = true;
            return true;
        }
        else{
            this.isLast = false;
            return false;
        }
    }

    public int[] getAnswer(){

        if(this.isFirst){
            return this.answer;
        }
        else{
            return this.prev.getAnswer();
        }
    }
    public double[][] getOutput() {
        return this.output;
    }
    public double[][] getFilters(){
        return this.weights;
    }

    public double[] getBias(){
        return this.bias;
    }


    public double[] getVbias() {
        return vbias;
    }

    public double[][] getVweights() {
        return vweights;
    }


    public void setVweights(double[][] vweights) {
        this.vweights = vweights;
    }

    public void setVbias(double[] vbias) {
        this.vbias = vbias;
    }


    public void setWindow(int s, int st){

            this.s = s;
            this.st = st;

    }


    //for validation set, adjust all arrays that are impacted by s, and st
    public void setInput(double[][][] input, int s, int st){
        this.setWindow(s, st);
        this.input = this.flatten(input, s, st);
        this.output = new double[this.input.length][this.weights.length];



    }

    public void setInput(double[][] input){
        this.input = input;
        this.output = new double[input.length][this.weights.length];
    }



}

