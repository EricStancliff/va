#include "VulkanInterface.h"
#include "VulkanInterfacePrivate.h"

VulkanInterface::VulkanInterface()
{
    m_p = std::make_unique<VulkanInterfacePrivate>();
}

void VulkanInterface::run()
{
    m_p->run();
}

VulkanInterface::~VulkanInterface()
{

}

