import torch
import torch.nn as nn
import torch.optim as optim
torch.set_default_device('cuda')


class GRUVAE(nn.Module):
    def __init__(
        self,
        input_size: int,   # dimension of each embedding at a timestep
        hidden_size: int,  # hidden dimension of GRU
        latent_size: int,  # dimension of latent space z
        num_layers: int = 1,
    ):
        super(GRUVAE, self).__init__()

        self.num_layers = num_layers
        self.latent_size = latent_size
        
        self.encoder_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.hidden2mu = nn.Linear(hidden_size, latent_size)
        self.hidden2logvar = nn.Linear(hidden_size, latent_size)
        
        self.latent2hidden = nn.Linear(latent_size, hidden_size)
        self.decoder_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.h2output = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        # We only need the final hidden state from the GRU
        _, h_n = self.encoder_gru(x)
        
        # h_n: (num_layers, batch_size, hidden_size)
        # Let's take only the top layer
        h_n_top = h_n[-1]  # shape: (batch_size, hidden_size)
        
        mu = self.hidden2mu(h_n_top)
        logvar = self.hidden2logvar(h_n_top)
        return mu, logvar

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        """
        Decodes a latent vector z into a sequence of length seq_len.
        Args:
            z: (batch_size, latent_size)
            seq_len: int, the length of the output sequence
        Returns:
            outputs: (batch_size, seq_len, input_size)
        """
        # Transform latent vector to initial hidden state for GRU
        hidden = self.latent2hidden(z)         # (batch_size, hidden_size)
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)  
        
        # We'll generate the sequence step by step.
        batch_size = z.size(0)
        outputs = []
        
        # Start with a zero vector as the "input" for each timestep
        input_step = torch.zeros(batch_size, 1, self.h2output.out_features, device=z.device)
        for t in range(seq_len):
            # Pass one step at a time
            out, hidden = self.decoder_gru(input_step, hidden)
            # out: (batch_size, 1, hidden_size)
            # Project back to embedding dimension
            step_output = self.h2output(out)   # (batch_size, 1, input_size)
            outputs.append(step_output)
            
            # The next input is the current output (autoregressive decoding)
            input_step = step_output

        # Concatenate along seq_len dimension
        outputs = torch.cat(outputs, dim=1)    # (batch_size, seq_len, input_size)
        return outputs

    def forward(self, x):

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        seq_len = x.size(1)
        recon_x = self.decode(z, seq_len)
        return recon_x, mu, logvar

    def sample(self, batch_size=1, seq_len=10):

        z = torch.randn(batch_size, self.latent_size).cuda()

        # Decode to generate sequences
        with torch.no_grad():
            samples = self.decode(z, seq_len)
        return samples