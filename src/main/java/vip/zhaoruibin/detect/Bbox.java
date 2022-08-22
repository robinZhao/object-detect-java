package vip.zhaoruibin.detect;
public class Bbox implements Cloneable{
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
		
		public Object clone() throws CloneNotSupportedException {
			return super.clone();
		}
	
	}