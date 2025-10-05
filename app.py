import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# ----------------------------------------------------------------------
# 1. ใช้ st.cache_resource เพื่อโหลดโมเดล YOLO เพียงครั้งเดียว (Best Practice)
# ----------------------------------------------------------------------

@st.cache_resource
def load_yolo_model(model_path):
    """โหลดโมเดล YOLO โดยใช้แคช"""
    try:
        # ใช้ชื่อไฟล์โมเดลที่ถูกต้อง 'yolo11n.pt'
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
        return None

# ----------------------------------------------------------------------
# 2. การตั้งค่า Streamlit และการรันแอปพลิเคชัน
# ----------------------------------------------------------------------

st.title("YOLO Image Detection App 📸")
st.markdown("---")

# โหลดโมเดล
model = load_yolo_model("yolo11n.pt") 

# Upload image
# แก้ไข: String literal ต้องจบอย่างถูกต้องในบรรทัดนี้
uploaded_image = st.file_uploader(
    "Upload an image (jpg, png) to run object detection", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    # ------------------------------------------------------------------
    # ส่วนการประมวลผลเมื่อมีไฟล์อัปโหลด
    # ------------------------------------------------------------------
    
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    
    # อ่านไฟล์ที่อัปโหลด
    image_data = uploaded_image.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image_np = np.array(image)
    
    st.info("Running YOLO object detection...")
    
    try:
        # รันการทำนายผลลัพธ์
        results = model.predict(image_np, conf=0.4, verbose=False)
        
        if results and len(results) > 0:
            result_image = results[0].plot()
            st.image(result_image, caption="YOLO Detection Result", use_container_width=True)
            st.success("Detection completed! 🎉")
        else:
            st.warning("Detection completed, but no objects were found with the given confidence.")
            
    except Exception as e:
        st.error(f"Error during detection: {e}")
        st.warning("Please check your model file or input image.")

st.markdown("---")
st.caption("Powered by Streamlit and Ultralytics YOLO.")
