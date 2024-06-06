package utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class OutFile {
    private final static String slash = "/";
    private PrintWriter outFile;
    private char delimit;

    public OutFile(final String outputPath, final String name) throws Exception {
        File dir = new File(outputPath);
        if (!dir.exists()) {
            boolean flag = dir.mkdirs();
            if (!flag) throw new Exception(outputPath + " not created!");
        }
        String filename = outputPath + name;
        if (!outputPath.endsWith(slash))
            filename = outputPath + slash + name;

        try {
            FileWriter fw = new FileWriter(filename);
            outFile = new PrintWriter(fw);
            delimit = ',';
        } catch (IOException exception) {
            System.err.println(exception + " File " + name + " Not found");
        }
    }

    public boolean writeString(String v) {
        outFile.print(v);
        return !outFile.checkError();
    }

    public boolean writeLine(String v) {
        outFile.print(v + "\n");
        return !outFile.checkError();
    }

    public boolean writeInt(int v) {
        outFile.print("" + v + delimit);
        return !outFile.checkError();
    }

    public boolean writeChar(char c) {
        outFile.print(c);
        return !outFile.checkError();
    }

    public boolean writeBoolean(boolean b) {
        outFile.print(b);
        return !outFile.checkError();
    }

    public boolean writeDouble(double v) {
        outFile.print("" + v + delimit);
        return !outFile.checkError();
    }

    public boolean newLine() {
        outFile.print("\n");
        return !outFile.checkError();
    }

    public void closeFile() {
        outFile.close();
    }
}
	