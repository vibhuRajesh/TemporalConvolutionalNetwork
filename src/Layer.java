import java.io.Serializable;

//Holder class for connecting everything
public abstract class Layer implements Serializable {
    private Layer next;
    private Layer prev;
    private boolean isFirst;
    private boolean isLast;

    public abstract void setNext(Layer prev);
    public abstract void setPrev(Layer next);
    public abstract Layer getPrev();
    public abstract Layer getNext();
    public abstract int getIndex();
    public abstract Object getOutput();
    public abstract boolean isFirst();
    public abstract boolean isLast();

    public static void connect(Layer one, Layer two){

        if(one.next == null){
            two.setPrev(one);
            one.setNext(two);
        }

        //functionality to insert between 2 layers
        // e.g. one.next != null
        else{
            Layer temp = one.getNext();
            two.setNext(temp);
            two.setPrev(one);
            one.setNext(two);
        }

        //update first/last
        one.isFirst();
        two.isFirst();

        //resets one if one was previous last
        one.isLast();
        two.isLast();


    }

    public abstract Object forward();
    public abstract void backward();

    public abstract int[] getAnswer();


}