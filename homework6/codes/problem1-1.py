import matplotlib.pyplot as plt

def read_sunspots(filepath):
    months = []
    values = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                m = int(float(parts[0]))
                v = float(parts[1])
            except ValueError:
                continue
            months.append(m)
            values.append(v)
    return months, values

def main(filepath):
    m,v = read_sunspots(filepath=filepath)
    plt.figure(figsize=(8,4))
    plt.plot(m,v)
    plt.xlabel("month")
    plt.ylabel("sunspots")
    plt.title("sunspots number vs month")
    plt.tight_layout()
    plt.savefig("./month-sunspots.png",dpi=300)
    plt.show()

if __name__ == "__main__":
    main(filepath='./codes/sunspots.txt')