package Game;

import Geom.Point3D;

import java.io.File;

public class Packman {

    private File packman;
    private int id;
    private double speed, x, y, z, radius;
    private Location location;
    //private Point3D location;



    public Packman(int id, double speed, Location startLocation, double radius){
        packman = new File("/packman.png");
        this.id = id;
        this.speed = speed;
        this.x = x;
        this.y = y;
        this.z = z;
        this.radius = radius;
        this.location = startLocation;
    }


    /*public Point3D getCurrentLocation() {
        return location;
    }

    public void setLocation(Point3D location) {
        this.location = location;
    }*/

    public Location getCurrentLocation() {
        return location;
    }

    public void setLocation(Location location) {
        this.location = location;
    }

    public int getId() {
        return id;
    }
}
