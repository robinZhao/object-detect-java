package vip.zhaoruibin.detect;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractDetectImg implements DetectImg{
	
	protected float[][][] inputArr;
	protected List<Bbox> boxesOutput=new ArrayList<Bbox>();
	protected List<Bbox> boxes=new ArrayList<Bbox>();
	protected Integer leftMargin;
	protected Integer topMargin;
	protected double r;
	protected long size;



	protected void unPadingAndunZoom(List<Bbox> boxes) {
		boxes.forEach(it -> {
			try {
				Bbox boxOrigImg = (Bbox) it.clone();
				boxOrigImg.x1 = (float) ((boxOrigImg.x1 - this.leftMargin) / r);
				boxOrigImg.y1 = (float) ((boxOrigImg.y1 - this.topMargin) / r);
				boxOrigImg.x2 = (float) ((boxOrigImg.x2 - this.leftMargin) / r);
				boxOrigImg.y2 = (float) ((boxOrigImg.y2 - this.topMargin) / r);
				this.boxes.add(boxOrigImg);
			} catch (CloneNotSupportedException e) {
				throw new RuntimeException(e);
			}
		});
	}
	

	public float[][][] getInputArr() {
		return inputArr;
	}

	
	public List<Bbox> getBoxes() {
		return boxes;
	}

	public List<Bbox> getBoxesOutput() {
		return boxesOutput;
	}

	public void setBoxesOutput(List<Bbox> boxesOutput) {
		this.boxesOutput = boxesOutput;
		unPadingAndunZoom(boxesOutput);
	}

	public void setBoxes(List<Bbox> boxes) {
		this.boxes = boxes;
	}
}
