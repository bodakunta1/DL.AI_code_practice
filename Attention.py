import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, d_model=2, row_dim=0, col_dim=1):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):

        # ✅ Q, K, V computation
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        print("\nQ:\n", q)
        print("\nK:\n", k)
        print("\nV:\n", v)

        # ✅ Similarity
        sims = torch.matmul(q, k.transpose(self.row_dim, self.col_dim))
        print("\nSimilarity Matrix:\n", sims)

        # ✅ Scaling
        scale = torch.tensor(k.size(self.col_dim), dtype=torch.float32) ** 0.5
        scaled_sims = sims / scale
        print("\nScaled Similarity:\n", scaled_sims)

        # ✅ Optional mask
        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
            print("\nMasked Scaled Similarity:\n", scaled_sims)

        # ✅ Softmax
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        print("\nAttention Percents:\n", attention_percents)

        # ✅ Output
        attention_scores = torch.matmul(attention_percents, v)
        print("\nFinal Attention Output:\n", attention_scores)

        return attention_scores


# =========================
# ✅ MAIN EXECUTION
# =========================
if __name__ == "__main__":

    # Encodings
    encodings_for_q = torch.tensor([
        [1.16, 0.23],
        [0.57, 1.36],
        [4.41, -2.16]
    ])

    encodings_for_k = torch.tensor([
        [1.16, 0.23],
        [0.57, 1.36],
        [4.41, -2.16]
    ])

    encodings_for_v = torch.tensor([
        [1.16, 0.23],
        [0.57, 1.36],
        [4.41, -2.16]
    ])

    # Seed for reproducibility
    torch.manual_seed(42)

    # ✅ Create Attention object (DO NOT overwrite class name)
    attention = Attention(d_model=2, row_dim=0, col_dim=1)

    # ✅ Run Attention
    output = attention(encodings_for_q, encodings_for_k, encodings_for_v)

    print("\n✅ FINAL OUTPUT FROM MAIN:\n", output)
