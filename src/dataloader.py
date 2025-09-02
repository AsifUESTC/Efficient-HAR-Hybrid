class HMDB51Dataset(Dataset):
    VIDEO_EXTENSION = '.avi'

    def __init__(self, df_hmdb51, transform=None, num_samples=16, frames_per_video=32, print_fps=False):
        if 'filename' not in df_hmdb51.columns or 'label' not in df_hmdb51.columns:
            raise ValueError("DataFrame must contain 'filename' and 'label' columns.")
        
        valid_videos = []
        for _, row in df_hmdb51.iterrows():
            video_path = row['filename'].replace('\\', '/')
            if self._is_valid_video(video_path):
                valid_videos.append({'filename': video_path, 'label': row['label']})

        if not valid_videos:
            raise ValueError("No valid videos found. Please check your dataset.")

        self.dataset = pd.DataFrame(valid_videos)
        self.video_paths = self.dataset['filename'].tolist()
        self.labels = self.dataset['label'].tolist()
        self.transform = transform
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.num_samples = num_samples
        self.frames_per_video = frames_per_video
        self.print_fps = print_fps

        self.vit_image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', do_rescale=False)

    def _is_valid_video(self, video_path):
        return os.path.isfile(video_path) and video_path.endswith(self.VIDEO_EXTENSION)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if self.print_fps:
            print(f"Video: {os.path.basename(video_path)}, FPS: {fps:.2f}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = self._get_frame_indices(total_frames)
        frames = []

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = Image.fromarray(frame)
                frame = self.transform(frame)
            frame = self.vit_image_processor(images=frame, return_tensors='pt')['pixel_values'].squeeze(0)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            print(f"Warning: No frames extracted from video: {video_path}")
            return torch.zeros((self.frames_per_video, 3, 224, 224)), 0

        frames = torch.stack(frames)
        label_index = self.label_to_index[label]
        return frames, label_index

    def _get_frame_indices(self, total_frames):
        if total_frames < self.frames_per_video:
            return [i % total_frames for i in range(self.frames_per_video)]
        step = total_frames // self.frames_per_video
        return [i * step for i in range(self.frames_per_video)]
