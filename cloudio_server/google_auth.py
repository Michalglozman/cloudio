import ee

should_auth = True

if (should_auth):
    ee.Authenticate()
    should_auth = False

