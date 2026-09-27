def format_idr(value):
    return "Rp{:,.0f}".format(value).replace(",", ".")


def bill_summary(rent, electricity, water, internet, other):
    values = {
        "Rent": float(rent),
        "Electricity": float(electricity),
        "Water": float(water),
        "Internet": float(internet),
        "Other": float(other),
    }

    total = sum(values.values())
    largest = max(values, key=values.get) if values else "None"

    return {
        "values": values,
        "total": total,
        "largest": largest,
    }
