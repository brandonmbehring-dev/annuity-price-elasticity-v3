#!/usr/bin/env python3
"""Test fixture data loading end-to-end."""

from pathlib import Path
from src.data.adapters import FixtureAdapter


def test_fixture_loading():
    """Test all fixture data sources load correctly."""
    fixtures_dir = Path('tests/fixtures/aws_complete')
    adapter = FixtureAdapter(fixtures_dir)

    print("Testing fixture data loading...")

    # Test 1: Sales data (don't filter by product)
    print("\n1. Loading sales data...")
    sales = adapter.load_sales_data(product_filter=None)  # No filter
    print(f"   ✓ Loaded {len(sales):,} total sales records")

    # Filter manually to verify FlexGuard data exists
    flexguard = sales[sales['product_name'].str.contains('FlexGuard', case=False, na=False)]
    print(f"   ✓ Found {len(flexguard):,} FlexGuard records")
    assert len(flexguard) > 0, "No FlexGuard data found"

    # Test 2: Competitive rates (CORRECT method name)
    print("\n2. Loading competitive rates...")
    rates = adapter.load_competitive_rates(start_date="2020-01-01")
    print(f"   ✓ Loaded {len(rates):,} rate records")
    assert len(rates) > 0, "Rates data is empty"

    # Test 3: Market weights (CORRECT method name)
    print("\n3. Loading market weights...")
    weights = adapter.load_market_weights()
    print(f"   ✓ Loaded {len(weights):,} weight records")
    assert len(weights) > 0, "Weights data is empty"

    # Test 4: Macro data (economic indicators from subdirectory)
    print("\n4. Loading macro data...")
    macro = adapter.load_macro_data()
    print(f"   ✓ Loaded {len(macro):,} macro records")
    assert len(macro) > 0, "Macro data is empty"

    print("\n✅ All fixture data sources loaded successfully!")
    return True


if __name__ == "__main__":
    test_fixture_loading()
