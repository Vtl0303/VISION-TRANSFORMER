"""
Deepfake Detection Desktop Application
Ứng dụng desktop nhận diện ảnh thật/fake sử dụng Vision Transformer
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkFont
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
import numpy as np
import warnings
from pathlib import Path
import sys
import threading

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR, LABELS_LIST

class DeepfakeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_variables()
        self.create_widgets()
        self.load_model()
        
    def setup_window(self):
        """Thiết lập cửa sổ chính"""
        self.root.title("🕵️ Deepfake Detection App")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        self.root.resizable(True, True)
        
        # Center window
        self.center_window()
        
    def center_window(self):
        """Căn giữa cửa sổ"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def setup_variables(self):
        """Thiết lập các biến"""
        self.model = None
        self.processor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_image = None
        self.prediction_result = None
        
    def create_widgets(self):
        """Tạo các widget"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🕵️ Deepfake Detection App", 
                               font=('Arial', 20, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Subtitle
        subtitle_label = ttk.Label(main_frame, text="Ứng dụng nhận diện ảnh thật/fake sử dụng Vision Transformer",
                                  font=('Arial', 10))
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Image selection
        left_frame = ttk.LabelFrame(main_frame, text="📤 Chọn ảnh", padding="15")
        left_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Image selection button
        self.select_btn = ttk.Button(left_frame, text="📁 Chọn ảnh từ máy tính", 
                                    command=self.select_image, width=25)
        self.select_btn.grid(row=0, column=0, pady=(0, 15))
        
        # Image preview
        self.image_label = ttk.Label(left_frame, text="Chưa có ảnh nào được chọn",
                                    background='white', relief='sunken', anchor='center')
        self.image_label.grid(row=1, column=0, pady=(0, 15), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Predict button
        self.predict_btn = ttk.Button(left_frame, text="🔍 Phân tích ảnh", 
                                     command=self.predict_image, state='disabled', width=25)
        self.predict_btn.grid(row=2, column=0, pady=(0, 10))
        
        # Image info
        self.info_text = tk.Text(left_frame, height=4, width=30, wrap=tk.WORD)
        self.info_text.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        # Right panel - Results
        right_frame = ttk.LabelFrame(main_frame, text="📊 Kết quả phân tích", padding="15")
        right_frame.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Result display
        self.result_frame = ttk.Frame(right_frame)
        self.result_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Default message
        self.default_label = ttk.Label(self.result_frame, 
                                      text="👆 Vui lòng chọn ảnh và click 'Phân tích ảnh' để xem kết quả",
                                      font=('Arial', 12), foreground='gray')
        self.default_label.grid(row=0, column=0, pady=50)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set(f"🖥️ Device: {self.device} | 📊 Classes: {', '.join(LABELS_LIST)}")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief='sunken')
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def load_model(self):
        """Load model trong thread riêng"""
        def load():
            try:
                self.status_var.set("🔄 Đang tải mô hình...")
                from transformers import ViTImageProcessor, ViTForImageClassification
                
                self.processor = ViTImageProcessor.from_pretrained(OUTPUT_DIR)
                self.model = ViTForImageClassification.from_pretrained(OUTPUT_DIR)
                self.model.to(self.device)
                self.model.eval()
                
                self.status_var.set(f"✅ Mô hình đã sẵn sàng! | Device: {self.device}")
                messagebox.showinfo("Thành công", "Mô hình đã được tải thành công!")
                
            except Exception as e:
                self.status_var.set("❌ Lỗi tải mô hình")
                messagebox.showerror("Lỗi", f"Không thể tải mô hình:\n{str(e)}")
        
        # Run in separate thread
        threading.Thread(target=load, daemon=True).start()
        
    def select_image(self):
        """Chọn ảnh từ máy tính"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh để phân tích",
            filetypes=file_types
        )
        
        if file_path:
            self.load_and_display_image(file_path)
            
    def load_and_display_image(self, file_path):
        """Load và hiển thị ảnh"""
        try:
            # Load image
            self.current_image = Image.open(file_path)
            
            # Convert to RGB if necessary
            if self.current_image.mode != 'RGB':
                self.current_image = self.current_image.convert('RGB')
            
            # Resize for display (max 300x300)
            display_image = self.current_image.copy()
            display_image.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(display_image)
            
            # Update image label
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            # Update info
            info_text = f"""📁 File: {Path(file_path).name}
📏 Kích thước: {self.current_image.size[0]} x {self.current_image.size[1]} pixels
🎨 Chế độ: {self.current_image.mode}
💾 Kích thước file: {Path(file_path).stat().st_size / 1024:.1f} KB"""
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info_text)
            
            # Enable predict button
            self.predict_btn.configure(state='normal')
            
            # Clear previous results
            self.clear_results()
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể đọc ảnh:\n{str(e)}")
            
    def predict_image(self):
        """Phân tích ảnh"""
        if self.current_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return
            
        if self.model is None or self.processor is None:
            messagebox.showwarning("Cảnh báo", "Mô hình chưa được tải!")
            return
            
        # Run prediction in separate thread
        def predict():
            try:
                self.status_var.set("🔄 Đang phân tích ảnh...")
                self.predict_btn.configure(state='disabled')
                
                # Preprocess image
                max_size = 512
                if max(self.current_image.size) > max_size:
                    ratio = max_size / max(self.current_image.size)
                    new_size = (int(self.current_image.size[0] * ratio), 
                               int(self.current_image.size[1] * ratio))
                    processed_image = self.current_image.resize(new_size, Image.Resampling.LANCZOS)
                else:
                    processed_image = self.current_image
                
                # Get model inputs
                inputs = self.processor(images=processed_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Make prediction
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
                
                # Get results
                predicted_class = int(np.argmax(probabilities))
                confidence = float(np.max(probabilities))
                class_name = LABELS_LIST[predicted_class]
                
                # Store result
                result = {
                    'prediction': class_name,
                    'confidence': confidence,
                    'probabilities': probabilities
                }
                
                # Update UI in main thread
                self.root.after(0, lambda: self.set_result_and_display(result))
                self.status_var.set("✅ Phân tích hoàn thành!")
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi khi phân tích:\n{str(e)}"))
                self.status_var.set("❌ Lỗi phân tích")
            finally:
                self.root.after(0, lambda: self.predict_btn.configure(state='normal'))
        
        threading.Thread(target=predict, daemon=True).start()
        
    def set_result_and_display(self, result):
        """Set result và hiển thị"""
        print(f"DEBUG: Setting result: {result}")  # Debug log
        self.prediction_result = result
        self.display_results()
        
    def display_results(self):
        """Hiển thị kết quả"""
        print(f"DEBUG: display_results called, prediction_result: {self.prediction_result}")
        
        if self.prediction_result is None:
            print("DEBUG: prediction_result is None, returning")
            return
            
        # Clear previous results
        self.clear_results()
        
        result = self.prediction_result
        if result is None:
            print("DEBUG: result is None after clear_results")
            return
            
        prediction = result.get('prediction', 'Unknown')
        confidence = result.get('confidence', 0.0)
        probabilities = result.get('probabilities', [0.5, 0.5])
        
        print(f"DEBUG: prediction={prediction}, confidence={confidence}, probabilities={probabilities}")
        
        # Result title
        if prediction == "Real":
            title_text = "✅ Kết quả: ẢNH THẬT"
            title_color = "#28a745"
            bg_color = "#d4edda"
        else:
            title_text = "❌ Kết quả: ẢNH FAKE"
            title_color = "#dc3545"
            bg_color = "#f8d7da"
            
        print(f"DEBUG: Creating title label: {title_text}")
        title_label = ttk.Label(self.result_frame, text=title_text, 
                               font=('Arial', 16, 'bold'), foreground=title_color)
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        print("DEBUG: Title label created and gridded")
        
        # Confidence
        conf_text = f"🎯 Độ tin cậy: {confidence:.1%}"
        conf_label = ttk.Label(self.result_frame, text=conf_text, 
                              font=('Arial', 14, 'bold'))
        conf_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Probabilities
        prob_frame = ttk.Frame(self.result_frame)
        prob_frame.grid(row=2, column=0, columnspan=2, pady=(0, 20))
        
        ttk.Label(prob_frame, text="📊 Xác suất chi tiết:", 
                 font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        for i, (label, prob) in enumerate(zip(LABELS_LIST, probabilities)):
            color = "#28a745" if label == "Real" else "#dc3545"
            
            # Label
            label_widget = ttk.Label(prob_frame, text=f"{label}:", 
                                    font=('Arial', 11, 'bold'), foreground=color)
            label_widget.grid(row=i+1, column=0, sticky=tk.W, padx=(0, 10))
            
            # Progress bar
            progress = ttk.Progressbar(prob_frame, length=200, mode='determinate')
            progress.grid(row=i+1, column=1, sticky=(tk.W, tk.E))
            progress['value'] = prob * 100
            
            # Percentage
            percent_label = ttk.Label(prob_frame, text=f"{prob:.1%}", 
                                     font=('Arial', 11, 'bold'), foreground=color)
            percent_label.grid(row=i+1, column=2, padx=(10, 0))
        
        # Action buttons
        button_frame = ttk.Frame(self.result_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(20, 0))
        
        ttk.Button(button_frame, text="🔄 Phân tích ảnh khác", 
                  command=self.clear_results).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="💾 Lưu kết quả", 
                  command=self.save_results).pack(side=tk.LEFT)
        
    def clear_results(self):
        """Xóa kết quả hiện tại"""
        print("DEBUG: clear_results called")
        for widget in self.result_frame.winfo_children():
            widget.destroy()
            
        # Don't reset prediction_result here, just clear the display
        # self.prediction_result = None  # Commented out this line
        
    def save_results(self):
        """Lưu kết quả ra file"""
        if self.prediction_result is None:
            messagebox.showwarning("Cảnh báo", "Không có kết quả để lưu!")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Lưu kết quả",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                result = self.prediction_result
                if result is None:
                    messagebox.showwarning("Cảnh báo", "Không có kết quả để lưu!")
                    return
                    
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("DEEPFAKE DETECTION RESULTS\n")
                    f.write("=" * 30 + "\n\n")
                    f.write(f"Prediction: {result.get('prediction', 'Unknown')}\n")
                    f.write(f"Confidence: {result.get('confidence', 0.0):.1%}\n\n")
                    f.write("Probabilities:\n")
                    probs = result.get('probabilities', [0.5, 0.5])
                    for label, prob in zip(LABELS_LIST, probs):
                        f.write(f"  {label}: {prob:.1%}\n")
                
                messagebox.showinfo("Thành công", f"Kết quả đã được lưu tại:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu file:\n{str(e)}")

def main():
    """Main function"""
    root = tk.Tk()
    app = DeepfakeDetectorApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 Đã thoát ứng dụng!")

if __name__ == "__main__":
    main()
