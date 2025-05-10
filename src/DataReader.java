import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;
import java.io.File;



public class DataReader {

    //Instance variables
        //file, column names, length
    private File f;
    private String[] colnames;
    private int length;

    //Values extracted from file
    private double[][] data;
    private String[] dates;

    //correct answers
    private int[] y;

    //correct answers for if folds are made
    private int[][] Y;


    /**
     * Constructor
     * @param filename: Assumed to be csv file structured in columns
     *                  FIRST COLUMN ASSUMED TO BE DATE
     */
    public DataReader(String filename){
        this.f = new File(filename);

        try {
            //use scanner to open file
            Scanner scanner = new Scanner(this.f);

            //first line of file has column names
            String first = scanner.nextLine();
            this.colnames = first.split(",");

            //find out length of file to instantiate array
            int lineCount = 0;
            while(scanner.hasNext()){
                lineCount++;
                String s = scanner.nextLine();
                String[] st = s.split(",");
            }
            this.length = lineCount;
            scanner.close();

            //rows = colnames - 1 (exclude date)
            //cols = essentially 'date'; just a transpose of the file for accessing 'columns' in one operation
            this.data = new double[this.colnames.length - 1][this.length];
            this.dates = new String[this.length];

            //create new scanner to actually access
            Scanner scanner2 = new Scanner(this.f);
            //get past column names
            scanner2.nextLine();
            int counter = 0;
            while(counter < this.length){
                //array of each row
                String s = scanner2.nextLine();
                String[] temp = s.split(",");

                this.dates[counter] = temp[0];
                //first column is always assumed to be dates
                for(int i = 1; i < temp.length; i++){
                    this.data[i - 1][counter] = Double.parseDouble(temp[i]);
                }
                counter++;
            }

        }

        catch(FileNotFoundException e){
            System.out.println(e.getMessage());
        }
    }


    /**
     * @param j which column to make a series out of
     *
     * @param p length of series (how many days of past to use in prediction)
     *           used in partitioning series
     *           reasoning: first: length - p is how many series of length p that are possible
     *                      last element is needed for label of last series
     * @return array of p length series
     */
    public double[][] makeSeries(int j, int p) {
        int numSeries = this.length - (p + 1);

        //row in data corresponding to jth column
        double[] temp = this.data[j];
        double[] copy = new double[temp.length];




        double[][] series = new double[numSeries][p];
        for (int i = 0; i < numSeries; i++) {
            for (int k = 0; k < p; k++) {
                //access jth element of temp, offset each time by ith iteration
                series[i][k] = temp[i + k];
            }
        }

        //instantiates y (correct answers) to have same length as number of samples.
        this.y = new int[series.length];
        //creates answers
        for (int i = 0; i < this.y.length; i++) {
            this.y[i] = 0;
            if (temp[i + p] > series[i][p - 1]) {
                this.y[i] = 1;
            }
        }


        return series;
    }

    /**
     * Overloaded method to make series and singular fold in one method using all columns but date
     * @param p: length of each series to make
     *
     * @param n: number of data points from the end to use
     *
     * @param a: column to base ground truth off of
     *
     * @return fold of n - (p + 1) series
     */
    public double[][][] makeSeries(int p, int n, int a){
        //number of dates is unchanged even when more input channel increases
        int numSeries = n - (p + 1);

        //column 0 is dates, so index beginning at 1
        double[][][] fold = new double[this.data.length][numSeries][p];
        //num columns
        for(int i = 0; i < fold.length; i++){
            //current column to 'series-ify'
            double[] temp = this.data[i];

            //starting index of where to begin 'fold'
            int start = temp.length - n;

            for(int j = 0; j < numSeries; j++){
                for(int k = 0; k < p; k++){
                    //making jth series with next p elements (k < p)
                    fold[i][j][k] = temp[start + j + k];

                }
            }

        }
        //instantiates y (correct answers) to have same length as number of samples.
        this.y = new int[numSeries];
        //creates answers
        //to get last n data points into series, start at
        int start = this.data[a].length - n;
        for (int i = 0; i < this.y.length; i++) {
            this.y[i] = 0;
            int last = p - 1;
            if (this.data[a][start + i + (last)] < this.data[a][(start) + i + (last + 1)]) {
                this.y[i] = 1;
            }
        }





        return fold;

    }


    //returns null if out of bounds
    public double[] getCol(int n){
        if(n > 0 && n < this.colnames.length - 1){
            return this.data[n];
        }
        return null;
    }
    //access columns by name
    //returns null if not found
    public double[] getCol(String s){
        for(int i = 0; i < this.colnames.length; i++){
            if(s.equals(this.colnames[i])){
                return this.getCol(i - 1);
            }
        }

        return null;
    }


    /**
     * @implNote Must have [series.length where series.length ≅ w mod t]
     * @param series 2D array of series of length p
     * @param w number of series in each fold
     * @param t size of test set of each fold (step size)
     * @return 3D array: think of as multiple `
     */
    public double[][][] makeFolds(double[][] series, int w, int t){
        int total = series.length;

        //first window of length w, so t doesn't start incrementing from 0, but w
        //so (total - w) / t = amount of steps taken for folds
        int i = (total - w) / t;

        //Trivially:
        //number of series in each fold
        int j = w + t;

        //width is length of each series
        int k = series[0].length;

        double[][][] temp = new double[i][j][k];
        this.Y = new int[i][j];

        //populate temp
        for(int l = 0; l < i; l++){
            for(int m = 0; m < j; m++){
                for(int n = 0; n < k; n++){
                    temp[l][m][n] = series[l*t + m][n];
                }
                this.Y[l][m] = this.y[l * t + m];
            }
        }

        System.out.println("num folds: " + i + ", window size: "+ j + ", series length: " + k);
        return temp;
    }



    //getter methods
    public int getLength(){
        return this.length;
    }

    public String[] getDates(){
        return this.dates;
    }

    public int[] y(){return this.y;}

    public int[][] Y(){return this.Y;}

}
