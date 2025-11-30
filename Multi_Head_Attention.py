import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# ✅ SINGLE ATTENTION HEAD
# =========================

class Attention(nn.Module):

    def __init__(self, d_model=2, row_dim=0, col_dim=1):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v):

        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        print("\nQ:\n", q)
        print("\nK:\n", k)
        print("\nV:\n", v)

        sims = torch.matmul(q, k.transpose(self.row_dim, self.col_dim))
        print("\nSimilarity Matrix:\n", sims)

        scale = torch.tensor(k.size(self.col_dim), dtype=torch.float32) ** 0.5
        scaled_sims = sims / scale
        print("\nScaled Similarity:\n", scaled_sims)

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        print("\nAttention Percents:\n", attention_percents)

        attention_scores = torch.matmul(attention_percents, v)
        print("\nFinal Attention Output (Single Head):\n", attention_scores)

        return attention_scores


# ==================================
# ✅ MULTI-HEAD ATTENTION
# ==================================

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model=2,
                 row_dim=0,
                 col_dim=1,
                 num_heads=1):

        super().__init__()

        # ✅ Create multiple attention heads
        self.heads = nn.ModuleList(
            [Attention(d_model, row_dim, col_dim)
             for _ in range(num_heads)]
        )

        self.col_dim = col_dim
        self.num_heads = num_heads

    def forward(self,
                encodings_for_q,
                encodings_for_k,
                encodings_for_v):

        print(f"\n==============================")
        print(f"Running Multi-Head Attention with {self.num_heads} heads")
        print(f"==============================")

        head_outputs = []

        for i, head in enumerate(self.heads):
            print(f"\n🔹 Head {i+1}")
            out = head(encodings_for_q,
                       encodings_for_k,
                       encodings_for_v)
            head_outputs.append(out)

        # ✅ Concatenate outputs from all heads
        output = torch.cat(head_outputs, dim=self.col_dim)

        print("\n✅ Final Multi-Head Output (After Concat):\n", output)

        return output


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

    # ====================================
    # ✅ TEST CASE 1: SINGLE HEAD
    # ====================================

    torch.manual_seed(42)

    print("\n\n==============================")
    print("✅ TESTING WITH 1 HEAD")
    print("==============================")

    multiHeadAttention = MultiHeadAttention(
        d_model=2,
        row_dim=0,
        col_dim=1,
        num_heads=1
    )

    output_1 = multiHeadAttention(
        encodings_for_q,
        encodings_for_k,
        encodings_for_v
    )

    print("\n✅ OUTPUT WITH 1 HEAD:\n", output_1)

    # ====================================
    # ✅ TEST CASE 2: TWO HEADS
    # ====================================

    torch.manual_seed(42)

    print("\n\n==============================")
    print("✅ TESTING WITH 2 HEADS")
    print("==============================")

    multiHeadAttention = MultiHeadAttention(
        d_model=2,
        row_dim=0,
        col_dim=1,
        num_heads=2
    )

    output_2 = multiHeadAttention(
        encodings_for_q,
        encodings_for_k,
        encodings_for_v
    )

    print("\n✅ OUTPUT WITH 2 HEADS:\n", output_2)
