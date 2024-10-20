import torch
import torchaudio
import torch.nn.functional as F

class AudioHead(torch.nn.Module):
    def __init__(self,):
        super().__init__()

        self.lstm1 = torch.nn.LSTM(768, 512, 3)
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(512, 512, 1, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Conv1d(512, 512, 1, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Conv1d(512, 512, 1, stride=1),
            torch.nn.ReLU(),

        )
        self.pooling = torch.nn.AvgPool1d(128)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.pooling(out.permute(0, 2, 1))
        return out.reshape(-1, 512)

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]

        return "".join([self.labels[i] for i in indices])

class MaskedAudioFeature(torch.nn.Module):
    def __init__(self, layer_id=12, device='cpu'):
        super().__init__()
        self.layer_id = layer_id
        self.device = device
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().to(device)
        self.blank = 0

    def forward(self, audio_file, frame_mask_len, out_len=128):
        cache_pth = audio_file.replace('.mp3', '.pt')

        try:
            features, indices = torch.load(cache_pth)
        except:
            waveform, sample_rate = torchaudio.load(audio_file)
            waveform = waveform.to(self.device)
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle.sample_rate)[0].reshape(1, -1)

            with torch.inference_mode():
                features, _ = self.model.extract_features(waveform, num_layers=self.layer_id+1)
                emission, _ = self.model(waveform)
                indices = torch.argmax(emission, dim=-1)

            # saving cache
            torch.save((features, indices), cache_pth)

        mask = (indices != self.blank).to(self.device)
        
        if features[self.layer_id].shape[1] > out_len:
            out = features[self.layer_id][:, :out_len]
        else:
            out = torch.zeros(out_len, 768)
            out[:features[self.layer_id].shape[1]] = features[self.layer_id][:features[self.layer_id].shape[1]]

        frame_mask = F.adaptive_avg_pool1d(mask.float(), frame_mask_len).reshape(-1).to(self.device)

        return out.reshape(-1, 768), frame_mask
    

if __name__ == '__main__':
    pass