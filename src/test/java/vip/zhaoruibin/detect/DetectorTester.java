package vip.zhaoruibin.detect;

import java.io.File;
import java.util.Arrays;

import org.junit.Test;

import ai.onnxruntime.OrtException;

public class DetectorTester {
	@Test
	public void testDetect() throws OrtException {
		OnnxImgDetector dector = new OnnxImgDetector();
		dector.loadOnnx("weights/lgt.onnx");
		File dir = new File("data/img");
		Arrays.stream(dir.listFiles()).forEach(it->{
			DetectImg img = dector.loadImg(DetectImgJdk.class,it.getAbsolutePath());
			try {
				dector.detect(img,0.25f,0.45f);
			} catch (OrtException e) {
				e.printStackTrace();
			}
			String[] nameArr = it.getName().split("\\.");
			img.drawOutputBoxes("data/output640/"+nameArr[0]+"-jdk."+nameArr[1]);
			img.drawBoxes("data/output/"+nameArr[0]+"-jdk."+nameArr[1]);
			img.getBoxes().forEach(b -> {
				System.out.println(String.format("%s\t%s\t%s\t%s\t%s\t%s", b.cls, b.score, b.x1, b.y1, b.x2, b.y2));
			});
		});
	}
	
	@Test
	public void testDetectOpencv() throws OrtException {
		OnnxImgDetector dector = new OnnxImgDetector();
		dector.loadOnnx("weights/lgt.onnx");
		File dir = new File("data/img");
		Arrays.stream(dir.listFiles()).forEach(it->{
			DetectImg img = dector.loadImg(DetectImgOpenCv.class,it.getAbsolutePath());
			try {
				dector.detect(img,0.25f,0.45f);
			} catch (OrtException e) {
				e.printStackTrace();
			}
			String[] nameArr = it.getName().split("\\.");
			img.drawOutputBoxes("data/output640/"+nameArr[0]+"-opencv."+nameArr[1]);
			img.drawBoxes("data/output/"+nameArr[0]+"-opencv."+nameArr[1]);
			img.getBoxes().forEach(b -> {
				System.out.println(String.format("%s\t%s\t%s\t%s\t%s\t%s", b.cls, b.score, b.x1, b.y1, b.x2, b.y2));
			});
		});
	}


}
