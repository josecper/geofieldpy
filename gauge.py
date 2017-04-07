def gauge(n, total):
    hashes = "#"*int(n*20/total)
    return f"[{hashes:<20}] {n}/{total} ({n/total*100:<3.1f}%)"
