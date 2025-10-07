# --- model loading ---
def load_models(self):
    try:
        self.status_label.config(text="Loading Autoencoder...")
        self.autoencoder_model = ConvAutoencoder().to(self.device)
        # robust load across PyTorch versions
        try:
            sd = torch.load(AUTOENCODER_MODEL_PATH, map_location=self.device, weights_only=True)
        except TypeError:
            sd = torch.load(AUTOENCODER_MODEL_PATH, map_location=self.device)
        self.autoencoder_model.load_state_dict(sd)
        self.autoencoder_model.eval()
        # small warmup
        _ = self.autoencoder_model(torch.zeros(1,3,IMAGE_RESIZE[1],IMAGE_RESIZE[0], device=self.device))
        print("Autoencoder loaded.")

        self.status_label.config(text="Loading YOLO...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        # warmup once (CPU is fine)
        _ = self.yolo_model(np.zeros((64,64,3), dtype=np.uint8), verbose=False)
        print("YOLO model loaded.")

        self.status_label.config(text="Models ready.")
        return True
    except Exception as e:
        self.status_label.config(text=f"Error loading models: {e}")
        print(f"Error during model loading: {e}")
        return False
