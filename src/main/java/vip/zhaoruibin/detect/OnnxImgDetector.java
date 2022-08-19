package vip.zhaoruibin.detect;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;
import ai.onnxruntime.TensorInfo;

public class OnnxImgDetector implements AutoCloseable {
	private static String[] hexs = new String[] { "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17",
			"3DDB86", "1A9334", "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085",
			"CB38FF", "FF95C8", "FF37C7" };
	private static Scalar[] colors;
	static {
		String path = "D:\\tools\\opencv\\build\\java\\x64\\opencv_java460.dll";
		System.load(path);
		colors = new Scalar[hexs.length];
		for (int i = 0; i < hexs.length; i++) {
			int r = Integer.parseInt(hexs[i].substring(0, 2), 16);
			int g = Integer.parseInt(hexs[i].substring(2, 4), 16);
			int b = Integer.parseInt(hexs[i].substring(4, 6), 16);
			colors[i] = new Scalar(b, g, r);
		}
	}

	public static class Bbox {
		String cls;
		int clsId;
		Float score;
		float x1;
		float y1;
		float x2;
		float y2;

		public Bbox(int clsId, String cls, Float score, float x1, float y1, float x2, float y2) {
			super();
			this.clsId = clsId;
			this.cls = cls;
			this.score = score;
			this.x1 = x1;
			this.y1 = y1;
			this.x2 = x2;
			this.y2 = y2;
		}
	}

	public static class DetectImg {
		private Mat image;
		Mat img_rgb;
		private float[][][] inputArr;
		private Integer leftMargin;
		private Integer topMargin;
		private List<Bbox> boxes;
		private double r;
		private long size;

		DetectImg(Mat image, long size, boolean auto) {
			this.image = image;
			this.size = size;
			this.initInputArr(auto);
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
			this.boxes.forEach(it -> {
				draw(this.image, it);
			});
			Imgcodecs.imwrite(outPath, this.image);
		}
	}

	public static Object[] max(float[] numberArray, int offset) {
		float max = numberArray[offset];
		int idx = offset;
		for (int i = offset + 1; i < numberArray.length; i++) {
			float score = numberArray[i];
			if (max < score) {
				max = score;
				idx = i;
			}
		}
		return new Object[] { idx - offset, max };
	}

	public static float iou(Bbox box1, Bbox box2) {
		if (box1 == null || box1 == null) {
			return 0.0f;
		}
		float ytop = Math.max(box1.y1, box2.y1);
		float ybottom = Math.min(box1.y2, box2.y2);
		float xleft = Math.max(box1.x1, box2.x1);
		float xright = Math.min(box1.x2, box2.x2);
		if (ybottom <= ytop || xright <= xleft) {
			return 0;
		}
		float inter = (ybottom - ytop) * (xright - xleft);
		return inter / ((box1.x2 - box1.x1) * (box1.y2 - box1.y1) + (box2.x2 - box2.x1) * (box2.y2 - box2.y1) - inter);
	}

	public static void draw(Mat img, Bbox box) {
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
	}

	private String[] models;
	private Integer stride;
	private OrtEnvironment env;
	private OrtSession session;
	private SessionOptions opts;
	private long imageSize;
	private String inputName;
	private Float iou_threshold = 0.45f;
	private Float conf_threshold = 0.25f;

	public OnnxImgDetector(float conf_threshold, float iou_threshold) {
		super();
		this.conf_threshold = conf_threshold;
		this.iou_threshold = iou_threshold;
	}

	public OnnxImgDetector() {
		this(0.25f, 0.45f);
	}

	public DetectImg loadImg(String imgPath, boolean auto) {
		return new DetectImg(Imgcodecs.imread(imgPath), this.imageSize, auto);
	}

	public DetectImg loadImg(String imgPath) {
		return loadImg(imgPath, true);
	}

	public void detect(DetectImg dectImg, float conf_threshold, float iou_threshold) throws OrtException {
		try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, new float[][][][] { dectImg.getInputArr() });
				Result output = session.run(Collections.singletonMap(inputName, inputTensor))) {
			OnnxTensor outputTensor = (OnnxTensor) output.get(0);
			List<Bbox> boxs = postProcess(outputTensor, conf_threshold, iou_threshold);
			boxs.forEach(it -> {
				draw(dectImg.img_rgb, it);
			});
			Imgcodecs.imwrite("out1.jpg", dectImg.img_rgb);
			dectImg.setBoxes(boxs);
		}
	}

	public void detect(DetectImg dectImg) throws OrtException {
		this.detect(dectImg, this.conf_threshold, this.iou_threshold);
	}

	public List<Bbox> detect(String imgPath, float conf_threshold, float iou_threshold) {
		DetectImg img = loadImg(imgPath);
		try {
			this.detect(img, conf_threshold, iou_threshold);
			return img.boxes;
		} catch (OrtException e) {
			throw new RuntimeException("detect failure", e);
		}
	}

	public List<Bbox> detect(String imgPath) {
		return this.detect(imgPath, this.conf_threshold, this.iou_threshold);
	}

	public List<Bbox> postProcess(OnnxTensor output, float conf_threshold, float iou_threshold) throws OrtException {
		AtomicInteger idx = new AtomicInteger(0);
		List<Bbox> bboxes = Arrays.stream(((float[][][]) output.getValue())[0]).filter(it -> it[4] > conf_threshold)
				.map(it -> {
					for (int j = 5; j < it.length; j++) {
						it[j] = it[j] * it[4];
					}
					Object[] max = max(it, 5);
					return new Object[] { it, max[0], max[1], null };
				}).filter(it -> (Float) it[2] > conf_threshold).map(it -> {
					float[] i = (float[]) it[0];
					return new Bbox((int) it[1], models[(int) it[1]], (Float) it[2], i[0] - i[2] / 2, i[1] - i[3] / 2,
							i[0] + i[2] / 2, i[1] + i[3] / 2);
				}).collect(Collectors.toList());

		return nms(bboxes, iou_threshold);
	}

	public List<Bbox> nms(List<Bbox> boxes, float iou_threshold) {
		boxes.sort(new Comparator<Bbox>() {
			@Override
			public int compare(Bbox o1, Bbox o2) {
				// TODO Auto-generated method stub
				return o2.score.compareTo(o1.score);
			}
		});

		LinkedList<Bbox> tmpList = new LinkedList(boxes);
		List<Bbox> keep_boxes = new ArrayList<>();
		while (tmpList.size() > 0) {
			Bbox box = tmpList.pop();
			keep_boxes.add(box);
			Iterator<Bbox> iter = tmpList.iterator();
			while (iter.hasNext()) {
				Bbox b = iter.next();
				if (iou(box, b) > iou_threshold) {
					iter.remove();
				}
			}
		}
		return keep_boxes;
	}

	public void loadOnnx(String onnxPath) throws OrtException {
		env = OrtEnvironment.getEnvironment();
		OrtSession.SessionOptions opts = new SessionOptions();
		opts.setOptimizationLevel(OptLevel.BASIC_OPT);
		session = env.createSession(onnxPath, opts);
		this.models = session.getMetadata().getCustomMetadataValue("names").map(it -> {
			return it.substring(it.indexOf('['), it.lastIndexOf(']'))
					.substring(it.indexOf('\'') + 1, it.lastIndexOf('\'')).split("'\\s*,\\s*'");
		}).orElse(new String[] {});
		this.stride = session.getMetadata().getCustomMetadataValue("stride").map(Integer::parseInt).orElse(32);
		inputName = session.getInputNames().iterator().next();
		NodeInfo info = session.getInputInfo().get(inputName);
		this.imageSize = (int) ((TensorInfo) info.getInfo()).getShape()[2];
	}

	@Override
	public void close() throws OrtException {
		this.session.close();
	}
}
