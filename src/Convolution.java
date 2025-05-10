import java.lang.Math;
import java.util.Arrays;
import java.util.Random;

public class Convolution extends Layer  {

    //inherited
    private Layer next;
    private Layer prev;
    private boolean isFirst;
    private boolean isLast;
    private boolean needsOne = false;

    //functionality
    private double learnRate;

    //Gradient descent with momentum
    //velocity tensors
    //backward pass
    private double[] vbias;
    private double[][] vOne;
    public double[][][] vfilters;

    private double rho;

    //start and stop window
    private int s;
    private int st;

    private int numFilters;

    //forward pass
    //dimensions: numFilters x input filters x kernel
    private double[][][] filters;
    private double[] bias;


    //filters' functionality
    private int kernel;
    private int dilation;

    private double[][] one;

    //used in backward pass
    private double[][][] prebias;
    private double[][][] preactivate;
    private double[][][] preone;
    //gradient to be passed back
    private double[][][] backs;

    //all for 1 window, e.g. input is 1 window, and answer is for 1 window
    private double[][][] input;
    private double[][][] output;
    private int[] answer;

    //access for residual blocks
    private double[][][] residual;
    private double[][][] adjResid;
    private boolean isResid;

    //layer to send back dResid
    Convolution reLayer;
    double[][][] dResid;
    private boolean sendsResid = false;



    /**
     * @implNote initializes filters using @this.randomizeFilters()
     * @param i: input for the layer
     *          takes in one window ( this.input ),
     * @param s: window start
     * @param st: window stop
     * @param n: number of filters to apply ( this.numFilters )
     * @param k: kernel size ( size of each filter, this.kernel )
     * @param d: dilation factor ( this.dilation )
     * @param r: if layer is a takes in a residual input from a previous layer
     * @param re: residual layer input (null if r is false)
     * @param l: learning rate ( scales gradient update, this.learnRate )
     **/
    public Convolution ( double[][][] i, int s, int st, int n, int k, int d, boolean r, Convolution re, double l ) {

        //direct input
        this.dilation = d;
        this.input = i;
        this.kernel = k;
        this.learnRate = l;
        this.numFilters = n;
        this.rho = 0.9;

        //numFilters: num output channels
        //used for dimensional compatibility
            //input.length = input channels
            //kernel = length of filter
        this.filters = new double[this.numFilters][this.input.length][this.kernel];
        //velocity tensor for filters, or doubles as gradient square tensor
        this.vfilters = new double[this.numFilters][this.input.length][this.kernel];

        //one bias per filter
        this.bias = new double[this.numFilters];
        //velocity tensor for bias
        this.vbias = new double[this.numFilters];

        //output and its 2 different copies
        this.output = new double[this.numFilters][i[0].length][i[0][0].length];
        this.prebias = new double[this.numFilters][i[0].length][i[0][0].length];
        this.preactivate = new double[this.numFilters][i[0].length][i[0][0].length];




        if (re != null) {
            this.residual = re.getInput();
            this.adjResid = new double[this.output.length][this.output[0].length][this.output[0][0].length];
            this.isResid = true;
            this.reLayer = re;
            this.preone = new double[this.residual.length][this.residual[0].length][i[0][0].length];

            //if residual layer and output's dimension are not same
            if (re.numFilters != this.numFilters) {
                this.needsOne = true;

                //input = depth, 1x1 h,w
                //numFilters = number of filters
                this.one = new double[this.numFilters][this.residual.length];

                //velocity tensor for 1x1 conv filters
                this.vOne = new double[this.numFilters][this.residual.length];

            }
            re.sendsResid(true);

        }

        else {
            this.residual = null;
            this.adjResid = null;
            this.isResid = false;
            this.reLayer = null;
        }


        this.s = s;
        this.st = st;

        //first layer channel depth = 1
        //only need to be called when initialized
        if(this.isFirst){
            this.normalize(this.input[0]);
        }

        this.randomizeFilters() ;
    }

    public double[][] diff(){
        return null;
    }
    //overloaded constructor for 2D input (first layer)
    /**
     * @param a: correct answers for current window (ground truth)
     */
    public Convolution ( double[][] i, int s, int st, int n, int k, int d, boolean r, Convolution re, double l, int[] a ) {

        //convert 2D array to 3D array
        double[][][] input = new double[1][i.length][i[0].length];
        input[0] = i;
        //use main constructor
        this(input, s, st, n, k, d, r, re, l);
        this.normalize(this.input[0]);

        this.answer = a;

    }

    //overloaded constructor for multichannel first layer input
    public Convolution ( double[][][] i, int s, int st, int n, int k, int d, boolean r, Convolution re, double l, int[] a ) {

        //convert 2D array to 3D array
        double[][][] input = (i);
        //use main constructor
        this(input, s, st, n, k, d, r, re, l);
        for(int it = 0; it < i.length; it++){
            this.normalize(this.input[it]);
        }
        this.answer = a;

    }


    //if dimensions are n x w x p:  changes w for val/test sets
    // only needs to update s and st for convolution, and FCN is automatically updated
    //take in 2D or 3D input
    public void setInput(double[][] input, int[] answer, int s, int st){
        double[][][] temp = new double[1][input.length][input[0].length];
        temp[0] = input;
        //called anytime new input is set (new window = new prices to normalize)
        this.setInput(temp, answer, s, st);

    }

    public void setInput(double[][][] input, int[] answer, int s, int st){
        this.s = s;
        this.st = st;

        this.input = input;
        this.normalize(this.input[0]);

        this.output = new double[this.numFilters][input[0].length][input[0][0].length];

        this.answer = answer;

        Layer iter = this.getNext();
        while(iter.getNext() != null){

            if(iter instanceof Convolution){
                ( (Convolution) iter ).setWindow(this.s, this.st);
            }

            else{
                if(iter.getPrev() instanceof Convolution){
                    ((FCN) iter).setInput(((Convolution) iter.getPrev()).getOutput(), s, st);
                }
                else{
                    ((FCN) iter).setInput(((FCN) iter.getPrev()).getOutput());
                }


                ((FCN) iter).setWindow(s, st);
            }

            iter = iter.getNext();
        }
        ((FCN) iter).setWindow(s, st);


    }

    public void normalize(double[][] arr){
        double sum = 0;

        //calculate mean
        for(int i = 0; i < arr.length; i++){
            for(int j = 0; j < arr[0].length; j++){
                sum += arr[i][j];
            }
        }

        double mean = sum / (arr.length * arr[0].length);

        //reset sum
        sum = 0;

        //calculate sd
        for(int k = 0; k < arr.length; k++){
            for(int l = 0; l < arr[0].length; l++){

                //(xi - mean)^2
                double diff = arr[k][l] - mean;
                sum += Math.pow(diff, 2);

            }
        }

        sum /= (arr.length * arr[0].length);
        double sd = Math.sqrt(sum);


        //actually normalizing
        for(int m = 0; m <  arr.length; m++){
            for(int n = 0; n < arr[0].length; n++){

                //z = (xi - u) / s
                arr[m][n] = (arr[m][n] - mean) / sd;
            }
        }

        //System.out.println(Arrays.deepToString(arr));

    }


    /**
     * @implNote sets main filter weights, bias and one by one convolution.
     */
    private void randomizeFilters() {
        //rv generator for filters, bias and one by one
        Random a = new Random() ;
        Random b = new Random() ;
        Random c = new Random() ;

        //window size for LeCun normal
        int length = (this.kernel * this.input.length);


        for( int h = 0; h < numFilters; h++ ) {
            for (int i = 0; i < this.filters[h].length; i++) {

                for (int j = 0; j < this.filters[h][i].length; j++) {

                    this.filters[h][i][j] = a.nextGaussian() *  (1.0 / ( (double) length) );
                    this.filters[h][i][j] *= Math.sqrt(length);
                }


            }

            this.bias[h] = b.nextGaussian() *  ( (1.0 / (double) length) );
            this.bias[h] *= Math.sqrt(length);

        }

        if(this.needsOne){
            for (int k = 0; k < this.one.length; k++) {
                for (int l = 0; l < this.one[0].length; l++) {

                    this.one[k][l] = c.nextGaussian() *  ( 1.0 / ( (double) length) );
                    this.one[k][l] *= Math.sqrt(length);
                }
            }
        }

    }

    /**
     *
     * @implNote method for biases, activation function, and one by one convolution
     * @implNote: essentially the forward pass:
     *      :stores output before bias and activation
     */
    public double[][][] forward() {

        if(!this.isFirst() ){

            //Convolution previous unlikely to be FCN
            if(this.prev instanceof Convolution){
                this.input = (double[][][]) this.prev.getOutput();
            }

            else if(this.prev instanceof FCN){

            }

        }

        this.convolve();

        //implemented in .forward() instead of constructor since needs to be updated with each forward pass
        this.prebias = this.copyArray(this.output);
        this.addBias() ;

        this.preactivate = this.copyArray(this.output);
        this.SeLU() ;


        //double check
        if ( this.isResid ) {
            this.preone = this.copyArray(this.residual);

            if ( this.needsOne ) {
                //residual input that is passed in (prior to 1x1 conv)
                this.one() ;
            }

            this.addResidual() ;
        }

        System.out.println("index " + this.getIndex() + "output" + (Arrays.toString(this.output[0][s])));

        return this.output;

    }


    /*
    FORWARD PASS METHODS
    */

    /**
     * @implNote updates this.output directly
     * @return 1 fold convolved ( 0 padding ) ( this.output )
     */
    public double[][][] convolve() {

        //each output channel (number of filters)
        for(int g = 0; g < this.numFilters; g++){

            //each series in window
            for(int h = s; h < st; h++){

                //each element in each training series
                for(int i = 0; i < this.input[0][0].length; i++) {

                    //sum resets whenever kernel moves to next element
                    double sum = 0;

                    //each element of kernel
                    for (int j = 0; j < this.kernel; j++) {

                        //number of input channels
                        for (int k = 0; k < this.input.length; k++) {

                            //zero padding
                            //kernel[length] = i = current spot in series
                            //(kernel - j+1) * dilation = first element in kernel
                            int index = i - ( ( this.kernel - ( j + 1) ) * this.dilation );
                            if ( index < 0 ) {
                                sum += 0;
                            }
                            else{
                                sum += this.input[k][h][index] * this.filters[g][k][j];
                            }

                        }

                    }
                    this.output[g][h][i] = sum;
                }
            }

        }

        return this.output;
    }


    /**
     * @implNote called in this.forward()
     */
    private void addBias() {

        for ( int h = 0; h < this.numFilters; h++ ) {
            for ( int i = this.s; i < this.st; i++ ) {
                for ( int j = 0; j < this.output[h][i].length; j++ ) {

                    //add bias by channel
                    this.output[h][i][j] += this.bias[h] * 0.01;
                }
            }
        }
    }

    /**
     * @implNote x<= 0: y = lambda * ( alpha * ( e^x - 1 ) ) , if x > 0: y = lambda * x
     * @implNote: alpha = 1.673263242354377284817042991671, lambda = 1.0507009873554804934193349852946 +
     * @implNote: in this.forward()
     */
    private void SeLU() {

        double alpha =  1.673263242354377284817042991671;
        double lambda = 1.0507009873554804934193349852946;

        for ( int h = 0; h < this.numFilters; h++ ) {
            for ( int i = this.s; i < this.st; i++ ) {
                for ( int j = 0; j < this.output[h][i].length; j++ ) {

                    //selu rule
                    if ( this.output[h][i][j] <= 0 ) {
                        this.output[h][i][j] = lambda * ( alpha * ( Math.exp ( this.output[h][i][j] ) - 1 ) ) ;
                    }

                    else{
                        this.output[h][i][j] = lambda * ( this.output[h][i][j] ) ;
                    }
                }
            }
        }

    }


    /**
     * @return one by one convolution
     * @implNote length of each filter is depth of residual
     *      1 x 1 x depth
     * @implNote: length of out is depth of numFilters
     */
    public double[][][] one() {

        //assumes residual window and output window are same size, as well as series length same
        double[][][] out = new double[this.numFilters][this.residual[0].length][this.residual[0][0].length];


        //each filter for num output filters (depth of output array)
        for(int g = 0; g < this.numFilters; g++) {

            //each training series
            for (int i = s; i < st; i++) {

                //each element of training series
                for (int j = 0; j < this.residual[0][0].length; j++) {

                    //sum resets with each column
                    double sum = 0;

                    //each element depth-wise in residual array
                    for (int h = 0; h < this.residual.length; h++) {
                        sum += this.residual[h][i][j] * this.one[g][h];
                    }

                    out[g][i][j] = sum;
                }

            }

        }
        //adjusted residual with correct dimensions for adding
        this.adjResid = out;
        return this.adjResid;

    }

    /**
     * @implNote: adds residual after check
     */
    private void addResidual() {
        if ( this.isResid ) {
            //add residual
            for (int h = 0; h < this.output.length; h++) {
                for(int i = s; i < st; i++){
                    for(int j = 0; j < this.output[h][i].length; j++){

                        //element wise addition
                        this.output[h][i][j] += this.adjResid[h][i][j];
                    }
                }
            }

        }
        //System.out.println("RESIDUAL OUTPUT"  + this.getIndex()  + " "  + Arrays.toString(this.output[0][0]));
    }

    /*
    BACKWARD PASS METHODS
    */

    /**
     * ADD SEED AND METHOD CALL
     * @implNote: perturbs 1 weight 1 time (either f(x+h) or f(x-h)), manually add method call in Network.train()
     * corresponding to the layer, then run 1 fwd pass
     * @param epsilon: perturbation value (negative for
     * @param idx: array with indices of weight to perturb
     *
     */
    public void gradCheck(double epsilon, int[] idx){
        int i = idx[0];
        int j = idx[1];
        int k = idx[2];
        this.filters[i][j][k] += epsilon;
    }

    /**
     * @implNote All elements of backward pass
     */
    public void backward(){

        //CASE I: NOT RESID
        //CASE II: SENDS/REC RESID
        //CASE IIa: RESID & 1X1
        //upstream gradient tensor to be sent through
        double[][][] grad;



        //getting upstream gradient tensor
        //.getBacks() method is slightly  different in FCN
        //only called one time in network
        grad = this.getBacks();

        //residual connection is added to output of layer so first in backpropagation
        if(this.isResid){

            //gradient of 1x1 filters
            if(this.needsOne) {
                this.dOne(grad);
            }

            //sets residual input layer's this.dResid as gradient to be sent back
            this.dResid(grad);
        }

        this.dSeLU(grad);

        this.dBias(grad);
        System.out.println("conv bias " + this.getIndex() + "\n" + Arrays.toString(this.dBias(grad)));


        this.dFilters(grad);
        System.out.println("conv filter " + this.getIndex() + "\n" + Arrays.toString(this.dFilters(grad)[0][0]));

        //adds and includes this.dResid if this.sendsResid is true
        this.dInput(grad);

    }


    /**
     * Values for L2 regularization to prevent overfitting
     */
    public double reg(){
        double sum = 0;

        for(int i = 0; i < this.filters.length; i++){
            for(int j = 0; j < this.filters[0].length; j++){
                for(int k = 0; k < this.filters[0][0].length; k++){
                    sum += this.filters[i][j][k] * this.filters[i][j][k];
                }
            }
        }

        return sum;

    }

    /**
     * @param u: upstream gradient: ∂L/∂g
     * @return 3D tensor of downstream gradients
     * @implNote e.g. if u = SeLU(e), calculates: ∂L/∂e = ∂L/∂u * ∂u/∂e
     */
    public double[][][] dSeLU(double[][][] u){
        double alpha = 1.673263242354377284817042991671;
        double lambda = 1.0507009873554804934193349852946;

        for(int i = 0; i < u.length; i++){
            for(int j = s; j < st; j++){
                for(int k = 0; k < u[0][0].length; k++){

                    //if (x > 0): ∂L/∂u *  λ
                    if(preactivate[i][j][k] > 0){
                        u[i][j][k] *= lambda;
                    }
                    else{
                        u[i][j][k] *= (lambda * alpha * Math.exp(preactivate[i][j][k]) );
                    }

                }
            }
        }


        return u;
    }


    /**
     * @implNote upstream gradient is typically tensor after dSeLU or dSigmoid
     * @implNote: updates biases as well
     * @param u: upstream gradient
     * @return gradient of bias terms
     */
    public double[] dBias(double[][][] u){

        double[] grad = new double[this.bias.length];

        //1 bias term per filter
        for(int i = 0; i < u.length; i++){
            double sum = 0;

            //sum across samples and series
            for(int j = s; j < st; j++){
                for(int k = 0; k < u[0][0].length; k++) {
                    sum += u[i][j][k];
                }
            }

            //gradient is summed across
            sum /= u[0].length * u[0][0].length;
            grad[i] = sum;

        }

        //update bias
        this.step(this.bias, grad, this.vbias, this.rho);

        return grad;
    }


    /**
     *
     * @param u: upstream gradient tensor, same as one after activation
     * @return 3D tensor of weights to update
     */
    public double[][][] dFilters(double[][][] u){

        //same dimensions as filters
        double[][][] grad = new double[this.numFilters][this.input.length][this.kernel];
        double sum = 0;

        //for each filter (and each output window)
        for(int i = 0; i < grad.length; i++){

            //for each set of filters (corresponding to input filters)
            for(int j = 0; j < grad[0].length; j++){

                //for each element of filter
                for(int k = 0; k < this.kernel; k++){
                    sum = 0;

                    //num series/length of upstream gradient window
                    for(int l = this.s; l < this.st; l++){

                        //length of series/width of upstream gradient
                        //for each element of upstream gradient
                        for(int m = 0; m < u[0][0].length; m++){

                            //i: current window
                            //l, m: length and width of window
                            double upstream = u[i][l][m];

                            //kernel[length] = i = current spot in series
                            //(kernel - j+1) * dilation = first element in kernel
                            int index = (m - ( (this.kernel -  (k + 1) ) * this.dilation) );

                            //if index is out of bounds (0 padding), then default is 0
                            //if in bounds, then local gradient is corresponding input coefficient
                            double local = 0;
                            if(index >= 0){
                                local = this.input[j][l][index];

                            }
                            sum += local * upstream;

                        }

                    }

                    grad[i][j][k] = sum;

                }
            }
        }
        this.step(this.filters, grad, this.vfilters, this.rho);

        return grad;
    }


    /**
     * @implNote dfor each local gradient, retraces filter's path because of causality
     * @param u: upstream gradient
     * @return gradient to be passed back to previous layer
     */
    public double[][][] dInput(double[][][] u){
        double[][][] grad = new double[this.input.length][this.input[0].length][this.input[0][0].length];
        double sum = 0;

        for(int i = 0; i < grad.length; i++){
            for(int j = this.s; j < this.st; j++){
                //for each individual element of downstream gradient
                for(int k = 0; k < grad[0][0].length; k++){
                    sum = 0;

                    //each output channel
                    for (int l = 0; l < this.numFilters; l++) {
                        //each element of upstream gradient series
                        for(int m = 0; m < this.input[0][0].length; m++) {
                            //because of causality, convolutions only affect values now or in past
                            if(m < k){
                                //do nothing ('future' datapoint)
                            }
                            else if(m == k){
                                //again by causality, last element of kernel is on present
                                sum += this.filters[l][i][this.kernel - 1] * u[l][j][m];
                            }
                            else{
                                //start iterating at end of kernel
                                int count = this.kernel - 1;
                                int it = m;
                                while(count >= 0 && it >= 0){
                                    if(it == k ){
                                        sum += this.filters[l][i][count] * u[l][j][m];
                                    }

                                    count--;
                                    it -= this.dilation;


                                }
                            }
                        }
                    }
                    grad[i][j][k] = sum;
                }
            }
        }

        //add residual gradient to final input tensor multivariable chain rule
        if(this.sendsResid){
            for (int n = 0; n < grad.length; n++) {
                for (int o = 0; o < grad[0].length; o++) {
                    for (int p = 0; p < grad[0][0].length; p++) {

                        grad[n][o][p] += this.dResid[n][o][p];

                    }
                }
            }
        }

        this.setBacks(grad);
        return grad;
    }

    /**
     * @implNote calculates gradient tensor to be sent back to this.re
     *      (residual input layer)
     * @param u: upstream gradient from next layer
     * @return downstream gradient to be sent to layer that sent residual
     */
    public double[][][] dResid(double[][][] u){
        //same dimensions as before one by one convolution
        double[][][] grad = new double[this.preone.length][this.preone[0].length][this.preone[0][0].length];
        double sum = 0;

        if(this.needsOne){//pre 1x1x conv number of channels
            for (int i = 0; i < grad.length; i++) {
                //window (batch elements
                for (int j = this.s; j < this.st; j++) {
                    //series length
                    for (int k = 0; k < grad[0][0].length; k++) {
                        sum = 0;

                        //number off output channels (local gradients)
                        for (int l = 0; l < this.output.length; l++) {
                            sum += u[l][j][k] * this.one[l][i];
                        }

                        grad[i][j][k] = sum;
                    }
                }
            }

        }
        else{
            grad = this.copyArray(u);
        }


        this.reLayer.setdResid(grad);
        return grad;
    }


    /**
     *
     * @param u: upstream gradient
     * @return gradients of 1x1 conv weights
     */
    public double[][] dOne(double[][][] u){
        double sum = 0;
        double[][] grad = new double[this.one.length][this.one[0].length];

        //for each 1x1 filter (number of output channels)
        for(int i = 0; i < this.one.length; i++){

            //number of channels in residual input
            for(int j = 0; j < this.one[0].length; j++){
                sum = 0;

                //in training sample window
                for(int k = this.s; k < this.st; k++){
                    //series length
                    for(int l = 0; l < this.input[0][0].length; l++){
                        sum += u[i][k][l] * this.preone[j][k][l];
                    }
                }

                grad[i][j] = sum;
            }
        }

        this.step(this.one, grad, this.vOne, this.rho);
        return grad;
    }

    //optimizer

    /**
     *
     * @param arr: tensor to be updated
     * @param grad: gradient tensor
     * @param velo: velocity/momentum tensor
     * @param rho: whether to use momentum (0 if not), also doubles as decay rate for RMSprop
     */

    public void step(double[] arr, double[] grad, double[] velo, double rho){


        for(int i = 0; i < grad.length; i++){

                //vt = vt-1 * ρ +( 1 - ρ ) * grad^2
                velo[i] = (rho * velo[i]) + ( (1 - rho) * grad[i] * grad[i]);

                //1E-8 for divide by 0 error
                arr[i] -= this.learnRate/Math.sqrt(velo[i] + 1E-8) * grad[i];

        }


    }
    public void step(double[][] arr, double[][] grad,double[][] velo, double rho){

        for(int i = 0; i < grad.length; i++){
            for(int j = 0; j < grad[0].length; j++){

                    //vt = vt-1 * ρ +( 1 - ρ ) * grad^2
                    velo[i][j] = (rho * velo[i][j]) + ( (1 - rho) * grad[i][j] * grad[i][j]);

                    //1E-8 for divide by 0 error
                    arr[i][j] -= this.learnRate/Math.sqrt(velo[i][j] + 1E-8) * grad[i][j];

            }
        }


    }
    public void step(double[][][] arr, double[][][] grad, double[][][] velo, double rho){

        for(int i = 0; i < grad.length; i++){
            for(int j = 0; j < grad[0].length; j++){
                for(int k = 0; k < grad[0][0].length; k++){

                    //vt = vt-1 * ρ +( 1 - ρ ) * grad^2
                    velo[i][j][k] = (rho * velo[i][j][k]) + ( (1 - rho) * grad[i][j][k] * grad[i][j][k]);

                    //1E-8 for divide by 0 error
                    arr[i][j][k] -= this.learnRate/Math.sqrt(velo[i][j][k] + 1E-8) * grad[i][j][k];

                }
            }
        }



//        //uses momentum (if rho is 0 then no momentum)
//        for(int j = 0; j < arr.length; j++) {
//            for (int k = 0; k < arr[0].length; k++) {
//                for (int l = 0; l < arr[0][0].length; l++) {
//                    //v =  rho * v + dw
//
//                    //w -= l * v
//                    velo[j][k][l] = velo[j][k][l] * rho + grad[j][k][l];
//                    arr[j][k][l] -=  this.learnRate * velo[j][k][l];
//
//                }
//            }
//        }

    }



    //copy 1, 2, and 3D array for manipulation
    public double[][][] copyArray(double[][][] arr){

        double[][][] temp = new double[arr.length][arr[0].length][arr[0][0].length];
        for(int i = 0; i < arr.length; i++){
            for(int j = 0; j<  arr[0].length; j++){
                for(int k = 0; k<  arr[0][0].length; k++){
                    temp[i][j][k] = arr[i][j][k];
                }
            }
        }
        return temp;
    }
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

    public void setRho(double rho){this.rho = rho;}

    //Getters and Setters for major variables

    public void setPrev ( Layer prev ) {
        if ( this.isFirst ) {
            this.prev = null;
        }
        else {
            this.prev = prev;
        }
    }
    public Layer getPrev() {
        if ( this.isFirst() ) {
            return null;
        }
        else {
            return this.prev;
        }
    }

    public void setNext( Layer next ) {
        this.next = next;
    }

    public Layer getNext() {
        if ( this.isLast() ) {
            return null;
        }
        else {
            return this.next;
        }
    }


    public void setFilters(double[][][] filters){this.filters = this.copyArray(filters);}

    public void setBias(double[] bias){this.bias = this.copyArray(bias);}


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


    public int getIndex() {
        if(this.prev == null){
            return 0;
        }
        else{
            return this.prev.getIndex() + 1;
        }

    }

    public void sendsResid(boolean val){this.sendsResid = val;}

    public void setdResid(double[][][] s){this.dResid = this.copyArray(s);}

    public double getLearnRate(){return this.learnRate;}


    //  filter/weight/output

    public double[][][] getFilters() {return this.filters;}
    public double[] getBias(){return this.bias;}
    public double[][][] getOutput() {return this.output;}



    public double[] getVbias() {return this.vbias;}

    public double[][] getvOne() {return this.vOne;}

    public double[][][] getVfilters() {return this.vfilters;}


    public void setVbias(double[] vbias) {this.vbias = this.copyArray(vbias);}

    public void setvOne(double[][] vOne) {this.vOne = this.copyArray(vOne);}

    public void setVfilters(double[][][] vfilters) {this.vfilters = this.copyArray(vfilters);}



    public double[][][] getInput(){return this.input;}

    public int[] getAnswer(){
        if(this.isFirst){
            return this.answer;
        }
        else{
            return this.prev.getAnswer();
        }
    }

    public double[][] getOne(){
        return this.one;
    }




    public boolean needsOne(){
        return this.needsOne;
    }


    public void setBacks(double[][][] backs){
        this.backs = this.copyArray(backs);
    }



    public double[][][] getBacks(){
        if(this.getNext() instanceof FCN){
            FCN temp = (FCN) this.getNext();
             return temp.getBacks();
        }
        else{
            return ( (Convolution) this.getNext() ) .backs;
        }
    }

    public int getNumFilters(){return this.numFilters;}
    public int getKernel(){return this.kernel;}

    public void setWindow(int s, int st){
        this.s = s;
        this.st = st;
    }
    public int getDilation(){return this.dilation;}

    public boolean isResid() {
        return this.isResid;
    }

    public void setOne(double[][] one){
        this.one = this.copyArray(one);
    }

    /**
     *
     * @return residual layer
     */
    public Convolution getResid(){return this.reLayer;}

    public int s(){return this.s;}

    public int st(){return this.st;}

}
