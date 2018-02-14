from unittest import TestCase
from IPython import embed
from peewee import *

db = PostgresqlDatabase(name='qlknn_test', database='qlknn_test')
class DatabaseTestCase(TestCase):
    database = db

    def setUp(self):
        if not self.database.is_closed():
            self.database.close()
        self.database.connect()
        super(DatabaseTestCase, self).setUp()

    def tearDown(self):
        super(DatabaseTestCase, self).tearDown()
        self.database.close()

    def execute(self, sql, params=None):
        return self.database.execute_sql(sql, params)


class ModelDatabaseTestCase(DatabaseTestCase):
    database = db
    requires = None

    def setUp(self):
        super(ModelDatabaseTestCase, self).setUp()
        self._db_mapping = {}
        # Override the model's database object with test db.
        if self.requires:
            for model in self.requires:
                self._db_mapping[model] = model._meta.database
                model._meta.set_database(self.database)

    def tearDown(self):
        # Restore the model's previous database object.
        if self.requires:
            for model in self.requires:
                model._meta.set_database(self._db_mapping[model])

        super(ModelDatabaseTestCase, self).tearDown()


class ModelTestCase(ModelDatabaseTestCase):
    database = db
    requires = None

    def setUp(self):
        super(ModelTestCase, self).setUp()
        if self.requires:
            self.database.drop_tables(self.requires, safe=True)
            self.database.create_tables(self.requires)

    def tearDown(self):
        # Restore the model's previous database object.
        try:
            if self.requires:
                self.database.drop_tables(self.requires, safe=True)
        finally:
            super(ModelTestCase, self).tearDown()


def requires_models(*models):
    def decorator(method):
        @wraps(method)
        def inner(self):
            _db_mapping = {}
            for model in models:
                _db_mapping[model] = model._meta.database
                model._meta.set_database(self.database)
            self.database.drop_tables(models, safe=True)
            self.database.create_tables(models)

            try:
                method(self)
            finally:
                try:
                    self.database.drop_tables(models)
                except:
                    pass
                for model in models:
                    model._meta.set_database(_db_mapping[model])
        return inner
    return decorator
