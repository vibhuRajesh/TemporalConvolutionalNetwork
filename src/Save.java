import java.io.*;

import java.util.ArrayList;

public class Save{

    /**
     *
     * @param network: Network to take in and serialize
     * @param folder: Network 'name' since file that is saved is always called "network.ser"
     */
    public static void saveNetwork(Network network, String folder){
        String folderPath = "src/" + folder;
        File p = new File(folderPath);

        //make folder if doesn't exist
        if(!p.exists()){
            if(p.mkdirs()){
                System.out.println("Created folder");
            }
            else{
                System.out.println("Failed");
            }

        }

        //else: if already exists, nothing needed
        String filePath =  folderPath + "/network.ser";
        try{

            FileOutputStream fos = new FileOutputStream(filePath);
            ObjectOutputStream oos = new ObjectOutputStream(fos);

            oos.writeObject(network);

            fos.close();
            oos.close();
        }

        catch(IOException i){
            System.out.println(i.getMessage());

            i.printStackTrace();
        }

    }

    /**
     *
     * @param folder: Network to instantiate
     * @return Deserialized network (class: Network)
     */
    public static Network loadNetwork(String folder){
        String filePath = "src/" + folder + "/network.ser";

        try{
            FileInputStream fis = new FileInputStream(filePath);
            ObjectInputStream ois = new ObjectInputStream(fis);

            Network obj = (Network) ois.readObject();

            fis.close();
            ois.close();

            return obj;
        }
        catch(IOException | ClassNotFoundException i){

            System.out.println("DIDN'T LOAD FILE");
            i.printStackTrace();

            return null;
        }
    }

}


