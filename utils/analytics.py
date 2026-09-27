from __future__ import annotations

def format_idr(value: float) -> str:
    return "Rp{:,.0f}".format(value).replace(",", ".")

def bill_summary(rent: float, electricity: float, water: float, internet: float, other: float) -> dict:
    values = {
        "Rent": float(rent),
        "Electricity": float(electricity),
        "Water": float(water),
        "Internet": float(internet),
        "Other": float(other),
    }
    total = sum(values.values())
    shares = {k: (v / total * 100 if total else 0.0) for k, v in values.items()}
    largest = max(values, key=values.get) if values else "None"
    return {"values": values, "total": total, "shares": shares, "largest": largest}
