import unittest

from build_region_partitions import alias_expression_identifiers


class AliasExpressionIdentifiersTest(unittest.TestCase):
    def test_aliases_only_available_feature_columns(self):
        expression = "(true) && ((pdgCode == 14 || pdgCode == -14))"

        aliased, aliases = alias_expression_identifiers(expression, {"pdgCode"})

        self.assertEqual(
            aliased,
            "(true) && ((feat__pdgCode == 14 || feat__pdgCode == -14))",
        )
        self.assertEqual(aliases, {"feat__pdgCode": "feat.pdgCode"})

    def test_does_not_alias_function_names_or_literals(self):
        expression = "true && int(count_scifi > 0)"

        aliased, aliases = alias_expression_identifiers(
            expression, {"count_scifi"}
        )

        self.assertEqual(aliased, "true && int(feat__count_scifi > 0)")
        self.assertEqual(
            aliases, {"feat__count_scifi": "feat.count_scifi"}
        )


if __name__ == "__main__":
    unittest.main()
