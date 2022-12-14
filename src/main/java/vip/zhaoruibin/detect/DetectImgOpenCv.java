package vip.zhaoruibin.detect;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class DetectImgOpenCv extends AbstractDetectImg {
	
	private static Scalar[] colors;
	static {
		Loader.load(opencv_java.class);
		colors = new Scalar[hexs.length];
		for (int i = 0; i < hexs.length; i++) {
			int r = Integer.parseInt(hexs[i].substring(0, 2), 16);
			int g = Integer.parseInt(hexs[i].substring(2, 4), 16);
			int b = Integer.parseInt(hexs[i].substring(4, 6), 16);
			colors[i] = new Scalar(b, g, r);
		}
	}

	private Mat image;
	Mat img_rgb;
	private String imgPath;

	public DetectImgOpenCv() {

	}

	public DetectImgOpenCv(String imgPath, long size, boolean auto) {
		this.load(imgPath, size, auto);
	}

	public void load(String imgPath, long size, boolean auto) {
		this.imgPath = imgPath;
		this.image = Imgcodecs.imread(imgPath);
		this.size = size;
		this.initInputArr(auto);
	}

	private Mat draw(Mat img, Bbox box) {
		Scalar borderColor = colors[box.clsId % colors.length];
		float x1 = box.x1;
		float y1 = box.y1;
		float width = (box.cls.length() + 5) * 11;
		if (x1 < 0) {
			x1 = x1 + (0 - x1);
		}
		if (y1 < 18) {
			y1 = y1 + (18 - y1);
		}
		if (x1 >= (img.width() - width)) {
			x1 = x1 - (img.width() - width);
		}
		Imgproc.rectangle(img, new Point(box.x1, box.y1), new Point(box.x2, box.y2), borderColor, 2);
		Imgproc.rectangle(img, new Point(x1, y1 - 18), new Point(x1 + (box.cls.length() + 5) * 11, y1), borderColor,
				-1);
		Imgproc.putText(img, box.cls + " " + String.format("%.2f", box.score), new Point(x1, y1 - 2),
				Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(255, 255, 255), 2);
		return img;
	}

	private void initInputArr(boolean auto) {
		double tmpR = ((double) size) / Math.max(image.width(), image.height());
		this.r = (!auto && tmpR > 1) ? 1 : tmpR;
		Mat img_rgb = new Mat();
		Mat img_resize = new Mat();
		this.img_rgb = img_rgb;
		if (r < 1 || (r > 1 && auto)) {
			Size destSize = new Size(Math.round(image.width() * r), Math.round(image.height() * r));
			Imgproc.resize(image, img_resize, destSize, r > 1 ? Imgproc.INTER_LINEAR : Imgproc.INTER_AREA);
		} else {
			img_resize = image;
		}
		Imgproc.cvtColor(img_resize, img_rgb, Imgproc.COLOR_BGR2RGB);
		this.leftMargin = (int) (size - img_resize.width()) / 2;
		this.topMargin = (int) (size - img_resize.height()) / 2;

		float[][][] arr = new float[3][(int) size][(int) size];
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (i >= topMargin && j >= leftMargin && i < topMargin + (int) img_resize.height()
						&& j < leftMargin + (int) img_resize.width()) {
					for (int k = 0; k < 3; k++) {
						arr[k][i][j] = (float) (img_rgb.get(i - topMargin, j - leftMargin)[k] / 255.0);
					}
				} else {
					arr[0][i][j] = 114.0f / 255;
					arr[1][i][j] = 114.0f / 255;
					arr[2][i][j] = 114.0f / 255;
				}
			}
		}
		this.inputArr = arr;
	}

	public void drawBoxes(String outPath) {
		Mat img = this.image.clone();
		this.boxes.forEach(it -> {
			draw(img, it);
		});
		Imgcodecs.imwrite(outPath, img);
	}

	@Override
	public void drawOutputBoxes(String outPath) {
		if (this.inputArr.length == 0) {
			throw new RuntimeException("no input arr data,can't draw output box");
		}
		Mat img = new Mat(new Size(size, size), CvType.CV_8UC3);
		byte[] destArr = new byte[(int) size * (int) size * 3];
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				for (int k = 0; k < 3; k++) {
					destArr[(i * (int) size + j) * 3 + k] = (byte) (inputArr[2 - k][i][j] * 255.0f);
				}
			}
		}
		img.put(0, 0, destArr);
		this.boxesOutput.forEach(it -> {
			draw(img, it);
		});
		Imgcodecs.imwrite(outPath, img);
	}
}