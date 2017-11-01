// Hideously ugly include order, has to be this way because of Magnum gl include wierdness. 

typedef unsigned char Byte;
#import <AppKit/NSApplication.h> // NSApplicationDelegate
#include "MeshTest.h"
#import <Cocoa/Cocoa.h>


@interface AppDelegate : NSObject <NSApplicationDelegate> {
    MeshTest *meshTest;
}

@property (assign, atomic) NSTimer *stepTimer; 
@property (assign, nonatomic) IBOutlet NSWindow *window;
@property (assign, nonatomic) IBOutlet NSSlider *areaSlider;
@property (assign, nonatomic) IBOutlet NSTextField *areaMin;
@property (assign, nonatomic) IBOutlet NSTextField *areaMax;
@property (assign, nonatomic) IBOutlet NSTextField *areaVal;

-(IBAction)run:(id)sender;

-(IBAction)step:(id)sender;

-(IBAction)stop:(id)sender;

-(IBAction)reset:(id)sender;

-(IBAction)areaValueChanged:(id)sender;

@end
