package Game;


import java.io.*;
import java.util.ArrayList;

public class Game {
    private ArrayList packmans;
    private ArrayList fruits;
    private File csvFile;
    private BufferedReader bufferedReader;


    public Game(File csvFile){
        if (csvFile.getPath().endsWith(".csv")){
            this.csvFile = csvFile;
        } else {
            throw new IllegalArgumentException();
        }
    }


    private void readKml(File csvFile){

    }

    public void readCsvFile() {
        try {
            bufferedReader = new BufferedReader(new FileReader(csvFile));
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                String[] data = line.split(",");
                Location location = new Location(Double.valueOf(data[2]), Double.valueOf(data[3]), Double.valueOf(data[4]));
                if (data[0].equals("P")){
                    packmans.add(new Packman(Integer.valueOf(data[1]), Double.valueOf(data[5]), location, Double.valueOf(data[6])));
                } else if (data[0].equals("F")) {
                    fruits.add(new Fruit(Integer.valueOf(data[1]), location, Double.valueOf(data[5])));
                } else {
                    throw new IllegalArgumentException();
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }




    public void saveGame2Kml() {

    }




}
