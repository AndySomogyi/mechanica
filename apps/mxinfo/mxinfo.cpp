#include <Corrade/PluginManager/Manager.h>
#include <Corrade/Utility/Arguments.h>
#include <Corrade/Utility/Directory.h>

#include "Magnum/Text/AbstractFont.h"
#include "Magnum/Text/AbstractFontConverter.h"
#include "Magnum/Text/DistanceFieldGlyphCache.h"
#include "Magnum/Trade/AbstractImageConverter.h"


#include <MxWindowless.h>


#include <iostream>

using namespace Magnum;
using namespace Magnum::Platform;

class MyApplication: public WindowlessApplication {
    public:
        MyApplication(const Arguments& arguments);

        int exec() override;
};

MyApplication::MyApplication(const Arguments& arguments):
    Platform::WindowlessApplication{arguments} {}

int MyApplication::exec() {
    Debug{} << "OpenGL version:" << GL::Context::current().versionString();
    Debug{} << "OpenGL renderer:" << GL::Context::current().rendererString();

    /* Exit with success */
    return 0;
}

/* main() function implementation */
MAGNUM_WINDOWLESSAPPLICATION_MAIN(MyApplication)
