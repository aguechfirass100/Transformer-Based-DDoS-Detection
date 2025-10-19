import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def explore_infinity_values(pkl_path: str):
    """Explore infinity values in the dataset"""

    print(f"Loading data from {pkl_path}")
    with open(pkl_path, 'rb') as f:
        flows_dict = pickle.load(f)

    # Convert to list for easier processing
    flows_data = []
    for flow_id, flow_info in flows_dict.items():
        flows_data.append({
            'flow_features': flow_info['flow_features'],
            'packets': flow_info['packets'],
            'label': flow_info['label']
        })

    print(f"Total flows: {len(flows_data)}")

    # Analyze flow features
    print("\n" + "=" * 50)
    print("FLOW FEATURES ANALYSIS")
    print("=" * 50)

    flow_features_list = [flow['flow_features'] for flow in flows_data]
    flow_df = pd.DataFrame(flow_features_list)

    print(f"Flow features shape: {flow_df.shape}")
    print(f"Flow feature columns: {flow_df.columns.tolist()}")

    # Check for infinity values in flow features
    infinity_summary = {}
    for col in flow_df.columns:
        try:
            numeric_col = pd.to_numeric(flow_df[col], errors='coerce')
            pos_inf = np.isposinf(numeric_col).sum()
            neg_inf = np.isneginf(numeric_col).sum()
            nan_count = numeric_col.isna().sum()

            if pos_inf > 0 or neg_inf > 0 or nan_count > 0:
                infinity_summary[col] = {
                    'pos_inf': pos_inf,
                    'neg_inf': neg_inf,
                    'nan': nan_count,
                    'finite_values': (~np.isnan(numeric_col) & np.isfinite(numeric_col)).sum(),
                    'total': len(numeric_col)
                }
        except:
            print(f"Could not analyze column {col} as numeric")

    if infinity_summary:
        print("\nColumns with infinity/NaN values:")
        for col, stats in infinity_summary.items():
            print(f"  {col}:")
            print(f"    +inf: {stats['pos_inf']} ({stats['pos_inf'] / stats['total'] * 100:.2f}%)")
            print(f"    -inf: {stats['neg_inf']} ({stats['neg_inf'] / stats['total'] * 100:.2f}%)")
            print(f"    NaN:  {stats['nan']} ({stats['nan'] / stats['total'] * 100:.2f}%)")
            print(f"    Finite: {stats['finite_values']} ({stats['finite_values'] / stats['total'] * 100:.2f}%)")
    else:
        print("No infinity or NaN values found in flow features")

    # Analyze packet features
    print("\n" + "=" * 50)
    print("PACKET FEATURES ANALYSIS")
    print("=" * 50)

    # Collect all packets
    all_packets = []
    for flow in flows_data:
        all_packets.extend(flow['packets'])

    if all_packets:
        packet_df = pd.DataFrame(all_packets)
        print(f"Packet features shape: {packet_df.shape}")
        print(f"Total packets: {len(all_packets)}")

        # Check for infinity values in packet features
        packet_infinity_summary = {}
        for col in packet_df.columns:
            try:
                numeric_col = pd.to_numeric(packet_df[col], errors='coerce')
                pos_inf = np.isposinf(numeric_col).sum()
                neg_inf = np.isneginf(numeric_col).sum()
                nan_count = numeric_col.isna().sum()

                if pos_inf > 0 or neg_inf > 0 or nan_count > 0:
                    packet_infinity_summary[col] = {
                        'pos_inf': pos_inf,
                        'neg_inf': neg_inf,
                        'nan': nan_count,
                        'finite_values': (~np.isnan(numeric_col) & np.isfinite(numeric_col)).sum(),
                        'total': len(numeric_col)
                    }
            except:
                print(f"Could not analyze packet column {col} as numeric")

        if packet_infinity_summary:
            print("\nPacket columns with infinity/NaN values:")
            for col, stats in packet_infinity_summary.items():
                print(f"  {col}:")
                print(f"    +inf: {stats['pos_inf']} ({stats['pos_inf'] / stats['total'] * 100:.2f}%)")
                print(f"    -inf: {stats['neg_inf']} ({stats['neg_inf'] / stats['total'] * 100:.2f}%)")
                print(f"    NaN:  {stats['nan']} ({stats['nan'] / stats['total'] * 100:.2f}%)")
                print(f"    Finite: {stats['finite_values']} ({stats['finite_values'] / stats['total'] * 100:.2f}%)")
        else:
            print("No infinity or NaN values found in packet features")

    # Label distribution
    print("\n" + "=" * 50)
    print("LABEL DISTRIBUTION")
    print("=" * 50)

    labels = [flow['label'] for flow in flows_data]
    label_counts = pd.Series(labels).value_counts()
    print(label_counts)

    # Plot label distribution
    plt.figure(figsize=(12, 6))
    label_counts.plot(kind='bar')
    plt.title('Label Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return infinity_summary, packet_infinity_summary


if __name__ == "__main__":
    # Adjust path to your data file
    pkl_path = "C:/Users/AGFirass/Documents/GitHub/Transformer-Based-DDoS-Detection/models/model_main/data/final_data/flows_dict.pkl"

    flow_inf, packet_inf = explore_infinity_values(pkl_path)