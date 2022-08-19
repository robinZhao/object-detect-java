package vip.zhaoruibin.detect;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;

class DetectImgJdk implements DetectImg {

	private static Color[] colors;
	static {
		colors = new Color[hexs.length];
		for (int i = 0; i < hexs.length; i++) {
			int r = Integer.parseInt(hexs[i].substring(0, 2), 16);
			int g = Integer.parseInt(hexs[i].substring(2, 4), 16);
			int b = Integer.parseInt(hexs[i].substring(4, 6), 16);
			colors[i] = new Color(r, g, b);
		}
	}

	private BufferedImage image;
	BufferedImage resizedImage;
	private float[][][] inputArr;
	private Integer leftMargin;
	private Integer topMargin;
	private List<Bbox> boxes;
	private double r;
	private long size;
	private String imgPath;
	private String imgFormat;

	public DetectImgJdk() {

	}

	public DetectImgJdk(String imgPath, long size, boolean auto) {
		this.load(imgPath, size, auto);
	}

	public void load(String imgPath, long size, boolean auto) {
		this.imgPath = imgPath;
		try {
			
			ImageInputStream s = ImageIO.createImageInputStream(new File(imgPath));
            Iterator<ImageReader> imageReaders = ImageIO.getImageReaders(s);
            while(imageReaders.hasNext()){
                ImageReader next = imageReaders.next();
                this.imgFormat = next.getFormatName();
                next.dispose();
                break;
            }
            this.image = ImageIO.read(s);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			throw new RuntimeException("read image failure", e);
		}
		this.size = size;
		this.initInputArr(auto);
	}

	private void draw(BufferedImage img, Bbox box) {
		Color borderColor = colors[box.clsId % colors.length];
		float x1 = box.x1;
		float y1 = box.y1;
		float width = (box.cls.length() + 5) * 11;
		if (x1 < 0) {
			x1 = x1 + (0 - x1);
		}
		if (y1 < 18) {
			y1 = y1 + (18 - y1);
		}
		if (x1 >= (img.getWidth() - width)) {
			x1 = x1 - (img.getWidth() - width);
		}
		Graphics2D g = img.createGraphics();
		g.setColor(borderColor);
		g.setStroke(new BasicStroke(3));
		g.drawRect((int) box.x1, (int) box.y1, (int) (box.x2 - box.x1), (int) (box.y2 - box.y1));
		g.fillRect((int) box.x1, (int) box.y1 - 18, (box.cls.length() + 5) * 11, 18);
		g.setColor(Color.WHITE);
		Font f = new Font(g.getFont().getFamily(),Font.BOLD,13);
		g.setFont(f);
		g.drawString(box.cls + " " + String.format("%.2f", box.score), (int) x1, (int) y1 - 2);
		g.dispose();

	}

	private void initInputArr(boolean auto) {
		double tmpR = ((double) size) / Math.max(image.getWidth(), image.getHeight());
		this.r = (!auto && tmpR > 1) ? 1 : tmpR;
		if (r < 1 || (r > 1 && auto)) {
			int destW = (int) Math.round(image.getWidth() * r);
			int destH = (int) Math.round(image.getHeight() * r);
			this.resizedImage=new BufferedImage(destW, destH, BufferedImage.TYPE_INT_RGB);
			Graphics g = resizedImage.createGraphics();
			g.drawImage(image.getScaledInstance(destW, destH, Image.SCALE_DEFAULT), 0, 0, null);
			g.dispose();
		} else {
			this.resizedImage = image;
		}
		int resizedWidth = resizedImage.getWidth();
		int resizedHeight = resizedImage.getHeight();
		this.leftMargin = (int) (size - resizedWidth) / 2;
		this.topMargin = (int) (size - resizedHeight) / 2;

		float[][][] arr = new float[3][(int) size][(int) size];
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (i >= topMargin && j >= leftMargin && i < topMargin + (int) resizedHeight
						&& j < leftMargin + (int) resizedWidth) {
					int rgb = resizedImage.getRGB( j - leftMargin,i - topMargin);
					Color color = new Color(rgb,true);
					arr[0][i][j]=color.getRed()/255.0f;
					arr[1][i][j]=color.getGreen()/255.0f;
					arr[2][i][j]=color.getBlue()/255.0f;
				} else {
					arr[0][i][j] = 114.0f / 255;
					arr[1][i][j] = 114.0f / 255;
					arr[2][i][j] = 114.0f / 255;
				}
			}
		}
		this.inputArr = arr;
	}

	public float[][][] getInputArr() {
		return inputArr;
	}

	private void unPadingAndunZoom(List<Bbox> boxes) {
		boxes.forEach(it -> {
			it.x1 = (float) ((it.x1 - this.leftMargin) / r);
			it.y1 = (float) ((it.y1 - this.topMargin) / r);
			it.x2 = (float) ((it.x2 - this.leftMargin) / r);
			it.y2 = (float) ((it.y2 - this.topMargin) / r);
		});
	}

	public List<Bbox> getBoxes() {
		return boxes;
	}

	public void setBoxes(List<Bbox> boxes) {
		unPadingAndunZoom(boxes);
		this.boxes = boxes;
	}

	public void drawBoxes(String outPath) {
		BufferedImage copy=new BufferedImage(image.getWidth(),image.getHeight(), BufferedImage.TYPE_INT_RGB);
		Graphics g = copy.createGraphics();
		g.drawImage(image, 0, 0, null);
		g.dispose();
		this.boxes.forEach(it -> {
			draw(copy, it);
		});
		try {
			ImageIO.write(copy, this.imgFormat, new File(outPath));
		} catch (IOException e) {
			throw new RuntimeException("write image failure",e);
		}
	}
}