from pathlib import Path
import rocksdb


class VariantDB:

    def __init__(self, path, rocksdb_options=None):
        if rocksdb_options is None:
            rocksdb_options = rocksdb.Options(
                create_if_missing=True,
                max_open_files=100,
            )

        self.db = rocksdb.DB(
            path,
            rocksdb_options,
            read_only=True
        )

    @staticmethod
    def _variant_to_byte(variant):
        return bytes(str(variant), 'utf-8')

    def _type(self, value):
        raise NotImplementedError()

    def _get(self, variant):
        if not variant.startswith('chr'):
            variant = 'chr%s' % variant
        return self.db.get(self._variant_to_byte(variant))

    def __getitem__(self, variant):
        maf = self._get(variant)
        if maf:
            return self._type(maf)
        else:
            raise KeyError('This variant is not in the db')

    def __contains__(self, variant):
        return self._get(variant) is not None

    def get(self, variant, default=None):
        try:
            return self[variant]
        except KeyError:
            return default


class VariantMafDB(VariantDB):

    def _type(self, value):
        return float(value)


