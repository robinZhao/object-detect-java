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
				dector.detect(img);
			} catch (OrtException e) {
				e.printStackTrace();
			}
			img.drawBoxes("data/output/"+it.getName());
			img.getBoxes().forEach(b -> {
				System.out.println(String.format("%s\t%s\t%s\t%s\t%s\t%s", b.cls, b.score, b.x1, b.y1, b.x2, b.y2));
			});
		});
		
	}

}
