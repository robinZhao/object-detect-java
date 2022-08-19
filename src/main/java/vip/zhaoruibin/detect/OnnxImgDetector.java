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

	public <T extends DetectImg>  T loadImg(Class<T> dectImgImplClass,String imgPath, boolean auto) {
		DetectImg img;
		try {
			img = dectImgImplClass.newInstance();
			img.load(imgPath, imageSize, auto);
			return (T)img;
		} catch (IllegalAccessException e) {
			throw new RuntimeException("Load Image failure",e);
		} catch (InstantiationException e) {
			throw new RuntimeException("Load Image failure",e);
		}
		
	}

	public <T extends DetectImg>  T loadImg(Class<T> dectImgImplClass,String imgPath) {
		return loadImg(dectImgImplClass,imgPath, true);
	}

	public void detect(DetectImg dectImg, float conf_threshold, float iou_threshold) throws OrtException {
		try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, new float[][][][] { dectImg.getInputArr() });
				Result output = session.run(Collections.singletonMap(inputName, inputTensor))) {
			OnnxTensor outputTensor = (OnnxTensor) output.get(0);
			List<Bbox> boxs = postProcess(outputTensor, conf_threshold, iou_threshold);
			dectImg.setBoxes(boxs);
		}
	}

	public void detect(DetectImg dectImg) throws OrtException {
		this.detect(dectImg, this.conf_threshold, this.iou_threshold);
	}

//	public List<Bbox> detect(String imgPath, float conf_threshold, float iou_threshold) {
//		DetectImg img = loadImg(imgPath);
//		try {
//			this.detect(img, conf_threshold, iou_threshold);
//			return img.boxes;
//		} catch (OrtException e) {
//			throw new RuntimeException("detect failure", e);
//		}
//	}

//	public List<Bbox> detect(String imgPath) {
//		return this.detect(imgPath, this.conf_threshold, this.iou_threshold);
//	}

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
