package Game;

import Geom.Point3D;

import java.io.File;

public class Fruit {

    private File fruit;
    private int id;
    //private Point3D location;
    private File packman;
    private double white;
    private Location location;

    public Fruit(int id, Location location, double white){
        this.id = id;
        fruit = new File("/fruit.png");
        this.location = location;
        this.white = white;

    }

    public Location getLocation() {
        return location;
    }

    public int getId() {
        return id;
    }

    public void setLocation(Location location) {
        this.location = location;
    }
}
