import csv

GROUPS = {
    'FC':        ['Plain-Old-CIFAR10-FC', 'D-shuffletruffle-FC', 'N-shuffletruffle-FC'],
    'CNN':       ['Plain-Old-CIFAR10-CNN', 'D-shuffletruffle-CNN', 'N-shuffletruffle-CNN'],
    'Attention': ['Plain-Old-CIFAR10-Attention', 'D-shuffletruffle-Attention', 'N-shuffletruffle-Attention'],
}

# Read test_summary.csv and keep only the last entry per model
with open('logs/test_summary.csv') as f:
    rows = list(csv.DictReader(f))

latest = {}
for row in rows:
    latest[row['model']] = row  # last occurrence wins

# Print table
W = [10, 29, 10, 14, 13]
SEP   = '├' + '┬'.join('─' * w for w in W) + '┤'
TOP   = '┌' + '┬'.join('─' * w for w in W) + '┐'
BOT   = '└' + '┴'.join('─' * w for w in W) + '┘'
DIV   = '├' + '┴'.join('─' * w for w in W) + '┤'
HDIV  = '├' + '┬'.join('─' * w for w in W) + '┤'

def row_str(cols):
    return '│' + '│'.join(f' {c:<{W[i]-2}} ' for i, c in enumerate(cols)) + '│'

def section_header(title):
    total = sum(W) + len(W) - 1
    return '│' + title.center(total) + '│'

print(TOP)
print(row_str(['Job', 'Model', 'Test Acc', 'Patch-16 Acc', 'Patch-8 Acc']))

for group, models in GROUPS.items():
    print(DIV)
    print(section_header(f'{group} Models'))
    print(HDIV)
    for i, model in enumerate(models):
        if model not in latest:
            continue
        r = latest[model]
        job = '-'
        cols = [
            job,
            model,
            f"{float(r['test_acc']):.2f}%",
            f"{float(r['patch16_acc']):.2f}%",
            f"{float(r['patch8_acc']):.2f}%",
        ]
        print(row_str(cols))
        if i < len(models) - 1:
            print(SEP)

print(BOT)
