"""
Tester script for quality_classify function.
"""

from cs336_data.utils import quality_classify


# Sample high-quality text (Wikipedia-style)
HIGH_QUALITY_TEXT = """
Lincoln's Greatest Speech - The Atlantic Skip to content Site Navigation • The Atlantic • PopularLatestNewsletters Sections • Politics • Ideas • Fiction • Technology • Science • Photo • Economy • Culture • Planet • Global • Books • AI Watchdog • Health • Education • Projects • 18000 Features • Family • Events • Washington Week • Progress • National Security • Explore The Atlantic Archive • games promo iconPlay The Atlantic Games • Listen to Podcasts and Articles The Print Edition View the current print edition Latest IssuePast Issues Give a Gift • Quick Links •
"""

# Sample low-quality text (spam/boilerplate-style)
LOW_QUALITY_TEXT = """
Site Maintenance We’ll be back soon! Sorry for the inconvenience but we’re performing some maintenance at the moment. If you need to you can always contact us, otherwise we’ll be back online shortly! — The Team
"""

MODEL_PATH = "quality_classifier.pt"


if __name__ == "__main__":
    print("Testing quality_classify function")
    print("=" * 50)

    print("\n[High Quality Text]")
    print(HIGH_QUALITY_TEXT[:100].strip() + "...")
    is_high, prob_high = quality_classify(HIGH_QUALITY_TEXT)
    print(f"Probability (positive): {prob_high:.4f}")
    print(f"Classification: {'positive' if prob_high >= 0.5 else 'negative'}")

    print("\n[Low Quality Text]")
    print(LOW_QUALITY_TEXT[:100].strip() + "...")
    is_high, prob_low = quality_classify(LOW_QUALITY_TEXT)
    print(f"Probability (positive): {prob_low:.4f}")
    print(f"Classification: {'positive' if prob_low >= 0.5 else 'negative'}")

    print("\n" + "=" * 50)
    print("Done.")
